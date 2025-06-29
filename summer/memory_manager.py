# 内置长期/短期记忆管理模块，兼容faiss
# 参考Letta设计，支持短期记忆、长期记忆、遗忘、召回
from collections import deque
import time, json, os
from summer.summer_faiss import faiss_add, faiss_recall, faiss_fuzzy_recall
from config import LOG_DIR
from config import WEIGHT_DECAY_DAYS, WEIGHT_DECAY_RATE, WEIGHT_DECAY_MIN, WEIGHT_DECAY_INTERVAL, MIN_WEIGHT_FORGET, MAX_UNUSED_DAYS_FORGET, REDUNDANCY_THRESHOLD
from config import SHORT_TERM_MEMORY_SIZE, LONG_TERM_CONSOLIDATE_WEIGHT, LONG_TERM_FORGET_DAYS
from summer.memory_flow.core_memory import CoreMemory # 核心记忆
from summer.memory_flow.archival_memory import ArchivalMemory # 归档记忆
from summer.memory_flow.short_term_memory import ShortTermMemory # 短期记忆
from summer.memory_flow.long_term_memory import LongTermMemory # 长期记忆
from concurrent.futures import ThreadPoolExecutor, as_completed # 并发工具

class MemoryManager:
    def __init__(self):
        self.core = CoreMemory() # 核心记忆
        self.archival = ArchivalMemory() # 归档记忆
        self.short_term = ShortTermMemory() # 短期记忆
        self.long_term_meta_path = f'{LOG_DIR}/faiss/faiss_metadata.json'
        self.long_term_usage_path = f'{LOG_DIR}/faiss/faiss_usage.json'
        self._load_long_term_meta()
        self.long_term = LongTermMemory(self.long_term_meta, self.long_term_usage) # 长期记忆

    def _load_long_term_meta(self):
        try:
            with open(self.long_term_meta_path, encoding='utf-8') as f:
                self.long_term_meta = json.load(f)
        except:
            self.long_term_meta = {}
        # 兼容旧数据，补全weight和last_used
        now = time.time()
        for k, v in self.long_term_meta.items():
            if 'weight' not in v:
                v['weight'] = 1
            if 'last_used' not in v:
                v['last_used'] = now
        try:
            with open(self.long_term_usage_path, encoding='utf-8') as f:
                self.long_term_usage = json.load(f)
        except:
            self.long_term_usage = {}

    def _ai_judge_important(self, chunk):
        """AI判定内容是否值得长期记忆，可用LLM/本地模型/规则等实现"""
        # 伪实现：如内容长度大于30字视为可能重要（可替换为LLM调用）
        return len(chunk.get('text', '')) > 30

    def add_memory(self, chunk, level='auto'):
        """Letta风格分层分流写入"""
        if level == 'auto':
            level = self._judge_level(chunk)
        if level == 'core':
            self.core.add(chunk)  # 核心记忆，极难遗忘
        elif level == 'archival':
            self.archival.add(chunk)  # 归档记忆，难遗忘
        elif level == 'long_term':
            self.long_term.add(chunk)  # 长期记忆，满足条件才写入faiss
        else:
            self.short_term.add(chunk)  # 短期记忆，仅内存

    def _judge_level(self, chunk):
        """LLM+规则自动分层判定，优先LLM，规则兜底"""
        text = chunk.get('text', '')
        # 1. LLM判定（可用本地或API）
        llm_type = self._llm_judge_type(text)
        if llm_type in ['core', 'archival', 'long_term', 'short_term']:
            return llm_type
        # 2. 规则兜底
        if any(kw in text for kw in ['我的名字', '身份', '出生', '联系方式']):
            return 'core'
        if any(kw in text for kw in ['重要事件', '历史', '关键时刻', '事故', '转折']):
            return 'archival'
        if len(text) > 30:
            return 'long_term'
        return 'short_term'

    def _llm_judge_type(self, text):
        """调用LLM判定内容分层，返回core/archival/long_term/short_term或None"""
        # 你可用本地模型或API，这里伪实现：如内容含"纠错"视为core
        if '纠错' in text:
            return 'core'
        # TODO: 替换为实际LLM API调用
        return None

    def recall_memory(self, query, k=5, levels=None, theme=None):
        """多层召回，异步并发检索，拼接高层内容，去重排序后返回k条"""
        levels = levels or ['core','archival','long_term','short_term'] # 默认层级
        results = [] # 结果列表
        futures = [] # future列表
        with ThreadPoolExecutor(max_workers=4) as executor: # 线程池
            for lvl in levels:
                if lvl == 'core':
                    futures.append(executor.submit(self.core.recall, query, k, theme)) # 核心记忆异步
                elif lvl == 'archival':
                    futures.append(executor.submit(self.archival.recall, query, k, theme)) # 归档记忆异步
                elif lvl == 'long_term':
                    # 长期记忆faiss召回本身返回future，需特殊处理
                    future = executor.submit(self.long_term.recall, query, k, theme) # 包一层，兼容接口
                    futures.append(future)
                elif lvl == 'short_term':
                    futures.append(executor.submit(self.short_term.recall, query, k, theme)) # 短期记忆异步
            for future in as_completed(futures): # 等待所有future完成
                try:
                    res = future.result()
                    if isinstance(res, list):
                        results += res
                except Exception as e:
                    pass # 忽略单层异常，保证主流程
        # 去重，按key唯一
        unique = {}
        for r in results:
            if 'key' in r:
                unique[r['key']] = r
        # 按权重和last_used排序
        sorted_results = sorted(unique.values(), key=lambda x: (-x.get('weight',1), -x.get('last_used',0)))
        return sorted_results[:k]

    def consolidate_memory(self):
        """将短期记忆中高权重内容巩固为长期记忆"""
        for chunk in list(self.short_term.memory):
            key = self._get_key(chunk)
            if self.long_term_meta.get(key, {}).get('weight', 0) >= LONG_TERM_CONSOLIDATE_WEIGHT:
                faiss_add([chunk])
                self.long_term_meta[key] = {'weight': self.long_term_meta.get(key, {}).get('weight', 1)}
                self.long_term_usage[key] = time.time()
        self._save_long_term_meta()

    def forget_long_term(self, min_weight=None, max_unused_days=None, redundancy_threshold=None):
        """
        长期记忆遗忘机制，定期清理权重低且长时间未用的内容，并物理重建faiss索引，同时去除冗余向量
        """
        import hashlib
        from summer.embedding import Emb
        from summer.faiss_index import FIndex
        import numpy as np

        now = time.time()
        faiss_dir = os.path.join(LOG_DIR, "faiss")
        log_msgs = []

        # 读取全局配置参数
        if min_weight is None:
            min_weight = MIN_WEIGHT_FORGET # 统一配置，长期记忆清理的最小权重
        if max_unused_days is None:
            max_unused_days = MAX_UNUSED_DAYS_FORGET # 统一配置，长期记忆清理的最大未用天数
        if redundancy_threshold is None:
            redundancy_threshold = REDUNDANCY_THRESHOLD # 统一配置，长期记忆冗余去重阈值

        important_deleted = []  # 记录被清理的人工标记内容
        # 读取单一faiss主库的meta和usage
        meta_path = os.path.join(faiss_dir, "faiss_metadata.json")
        usage_path = os.path.join(faiss_dir, "faiss_usage.json")
        idx_path = os.path.join(faiss_dir, "faiss.index")

        # 加载meta和usage
        try:
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
        except:
            meta = {}
        try:
            with open(usage_path, encoding="utf-8") as f:
                usage = json.load(f)
        except:
            usage = {}

        # 1. 找出需要遗忘的key（权重低且长时间未用，优先保留高权重）
        to_delete = set()
        for k, v in meta.items():
            last_used = usage.get(k, 0)
            weight = v.get("weight", 1)
            # 人工标记important的内容，只有一年未用才会被清理
            if v.get('important', False):
                if now - last_used <= 365 * 86400:
                    continue  # 一年内未用，永不清理
            # 只有权重低且长时间未用的才会被遗忘
            if weight <= min_weight and now - last_used > max_unused_days * 86400:
                to_delete.add(k)
                if v.get('important', False):
                    important_deleted.append(k)  # 记录被清理的人工标记内容

        # 2. 收集未被遗忘的文本
        keep_chunks = []
        for k in meta:
            if k in to_delete:
                continue
            for fn in os.listdir(LOG_DIR):
                if not fn.endswith('.txt'):
                    continue
                with open(f'{LOG_DIR}/{fn}', encoding='utf-8') as r:
                    t = None
                    for l in r:
                        if l.strip().startswith('时间:'):
                            t = l.split(':', 1)[1].strip()
                        for role in ['用户', 'user', '娜迦', 'ai']:
                            if l.strip().startswith(f'{role}:'):
                                txt = l.split(':', 1)[1].strip()
                                ck = hashlib.md5(f'{fn}_{t}_{role}_{txt}'.encode()).hexdigest()
                                if ck == k:
                                    keep_chunks.append({'role': 'user' if '用户' in role else 'ai', 'text': txt, 'time': t, 'file': fn, 'key': ck})

        # 3. 冗余向量检测（高相似度去重）
        del_dup = set()
        if keep_chunks:
            vecs = Emb().enc([c['text'] for c in keep_chunks])
            for i in range(len(vecs)):
                for j in range(i+1, len(vecs)):
                    if np.dot(vecs[i], vecs[j])/(np.linalg.norm(vecs[i])*np.linalg.norm(vecs[j])) > redundancy_threshold:
                        del_dup.add(keep_chunks[j]['key'])
        # 合并所有要删除的key
        remove = to_delete | del_dup
        keep = [c for c in keep_chunks if c['key'] not in remove]

        # 4. 物理重建faiss索引
        if keep:
            vs = Emb().enc([c['text'] for c in keep])
            fidx = FIndex()
            fidx.add(vs)
            fidx.save(idx_path)
        else:
            if os.path.exists(idx_path):
                os.remove(idx_path)

        # 5. 更新meta和usage
        for k in remove:
            meta.pop(k, None)
            usage.pop(k, None)
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=0, indent=1)
        with open(usage_path, 'w', encoding='utf-8') as f:
            json.dump(usage, f, ensure_ascii=0, indent=1)

        log_msg = f"夏园系统清理{len(to_delete)}条，冗余去重{len(del_dup)}条，保留{len(meta)}条。"
        if important_deleted:
            log_msg += f" [警告] 清理了{len(important_deleted)}条人工标记重要内容: {important_deleted}"
        log_msgs.append(log_msg)

        print("[记忆遗忘] " + " ".join(log_msgs))

    def _save_long_term_meta(self):
        with open(self.long_term_meta_path, 'w', encoding='utf-8') as f:
            json.dump(self.long_term_meta, f, ensure_ascii=0, indent=1)
        with open(self.long_term_usage_path, 'w', encoding='utf-8') as f:
            json.dump(self.long_term_usage, f, ensure_ascii=0, indent=1)

    def _get_key(self, chunk):
        import hashlib
        return hashlib.md5(f"{chunk.get('file','')}_{chunk.get('time','')}_{chunk.get('role','')}_{chunk['text']}".encode()).hexdigest()

    def _decay_weights(self, decay_days=WEIGHT_DECAY_DAYS, decay_rate=WEIGHT_DECAY_RATE, min_weight=WEIGHT_DECAY_MIN):
        """权重衰减：长时间未用自动降低权重，高权重记忆衰减更慢"""
        now = time.time()
        for k, v in self.long_term_meta.items():
            last_used = v.get('last_used', now)
            days_unused = (now - last_used) / 86400
            if days_unused > decay_days and v.get('weight', 1) > min_weight:
                # 衰减公式：每decay_days未用，权重乘以decay_rate
                v['weight'] = max(int(v['weight'] * decay_rate), min_weight)
        self._save_long_term_meta()

    def build_context(self, user_input, k=5, max_tokens=1500):
        """自动上下文拼接，优先拼接高层记忆，token超限优先保留高层"""
        import tiktoken
        context_lines = []
        total_tokens = 0
        enc = tiktoken.encoding_for_model('gpt-3.5-turbo') if hasattr(tiktoken,'encoding_for_model') else lambda x: x.split()
        for lvl in ['core','archival','long_term','short_term']:
            mems = self.recall_memory(user_input, k, levels=[lvl])
            for c in mems:
                line = f"[历史][{c.get('time','')}][{c.get('role','')}]:{c.get('text','')}"
                tokens = len(enc(line))
                if total_tokens + tokens > max_tokens:
                    return "\n".join(context_lines)
                context_lines.append(line)
                total_tokens += tokens
        return "\n".join(context_lines)

    def mark_important(self, key, delta=10):
        """人工/AI标记高价值记忆，权重大幅提升，防止被遗忘"""
        if key in self.long_term_meta:
            self.long_term_meta[key]['weight'] = self.long_term_meta[key].get('weight', 1) + delta
            self._save_long_term_meta()

    def mark_important_batch(self, keys, delta=10):
        """批量标记高价值记忆，提升多条记忆权重"""
        updated = 0
        for key in keys:
            if key in self.long_term_meta:
                self.long_term_meta[key]['weight'] = self.long_term_meta[key].get('weight', 1) + delta
                updated += 1
        if updated:
            self._save_long_term_meta()
        return updated

    def adjust_weights_periodically(self, interval=WEIGHT_DECAY_INTERVAL):
        """每interval轮对话批量衰减权重"""
        if not hasattr(self, '_adjust_counter'):
            self._adjust_counter = 0
        self._adjust_counter += 1
        if self._adjust_counter >= interval:
            self._decay_weights()
            self._adjust_counter = 0

    def fuzzy_recall(self, query, k=5):
        """模糊召回方法，兼容conversation_core.py的调用"""
        try:
            # 使用faiss_fuzzy_recall进行模糊搜索
            return faiss_fuzzy_recall(query, k)
        except Exception as e:
            # 降级到普通召回
            return self.recall_memory(query, k)

    def _is_core(self, chunk):
        # 可扩展LLM/规则判定核心记忆
        return chunk.get('level') == 'core' or '身份' in chunk.get('text','')
    def _is_archival(self, chunk):
        # 可扩展LLM/规则判定归档记忆
        return chunk.get('level') == 'archival' or '重要事件' in chunk.get('text','')
