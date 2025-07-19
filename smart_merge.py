import os
import re
import json
import subprocess
import logging
import sys
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

# --- 配置 ---
# 配置日志系统，将日志输出到标准错误流，避免干扰结果
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stderr)

# --- 模型和API配置 ---
# 使用用户直接提供的API Key
DASHSCOPE_API_KEY = "sk-480377f39a564d67b63def422ccb52cc"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_NAME = "qwen-max" # 更新为通义千问Max，以获得更好的JSON输出和逻辑处理能力

# --- LLM客户端初始化 ---
# 由于API Key已硬编码，不再需要检查环境变量
try:
    client = OpenAI(
        api_key=DASHSCOPE_API_KEY,
        base_url=BASE_URL,
    )
except Exception as e:
    logging.error(f"初始化OpenAI客户端时出错: {e}")
    sys.exit(1)

# --- 核心功能函数 ---

def get_conflicting_files():
    """使用 git grep 查找所有包含冲突标记的文件。"""
    logging.info("正在查找包含冲突标记的文件...")
    try:
        result = subprocess.run(
            ['git', 'grep', '-l', '<<<<<<< HEAD'],
            check=True,
            capture_output=True,
            text=True,
            encoding='utf-8'
        )
        files = result.stdout.strip().split('\n')
        return [f for f in files if f]
    except subprocess.CalledProcessError:
        logging.info("没有找到包含冲突标记的文件。")
        return []
    except Exception as e:
        logging.error(f"查找冲突文件时发生未知错误: {e}")
        return []

def resolve_conflict_chunk(ours_chunk, theirs_chunk):
    """将单个冲突块发送给LLM进行分析、分类和合并。"""
    prompt = f"""You are an expert programmer tasked with resolving a Git merge conflict.
Analyze the 'ours' and 'theirs' code blocks and classify the conflict into one of three types: 'identical', 'one-sided', or 'complex'.
Then, provide the correctly merged code.

Your response MUST be a single, valid JSON object with two keys: "conflict_type" and "merged_code".

Conflict Types Definitions:
1. `identical`: The two blocks are functionally and semantically identical. Whitespace or minor comment differences are acceptable. The merged code should be one of the blocks.
2. `one-sided`: One block contains all the logic of the other, plus new additions or clear improvements. The merge is a simple choice of the more advanced block.
3. `complex`: Both blocks contain significant, independent changes that need to be carefully integrated to preserve both sets of logic.

'Ours' changes (this is from the branch being merged into, likely with newer features):
---
{ours_chunk}
---

'Theirs' changes (this is from the branch being merged, likely the main branch):
---
{theirs_chunk}
---

Now, provide the JSON object:"""

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{'role': 'user', 'content': prompt}],
            temperature=0.0,
            response_format={"type": "json_object"}, # 开启JSON模式
        )
        response_text = completion.choices[0].message.content.strip()
        response_json = json.loads(response_text)
        
        # 验证返回的JSON结构
        if "conflict_type" in response_json and "merged_code" in response_json:
            return response_json
        else:
            raise ValueError("LLM返回的JSON缺少必要键。")

    except Exception as e:
        logging.error(f"调用LLM API或解析JSON时出错: {e}")
        return {
            "conflict_type": "failed",
            "merged_code": f"<<<<<<< HEAD\n{ours_chunk}\n=======\n{theirs_chunk}\n>>>>>>> auto-merge-failed"
        }

def process_file(filepath):
    """处理单个文件：读取、查找冲突、调用AI解决、写回，并返回处理结果。"""
    logging.info(f"正在处理文件: {filepath}")
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except Exception as e:
        logging.error(f"无法读取文件 {filepath}: {e}")
        return {"filepath": filepath, "status": "read_error", "conflict_types": []}

    conflict_pattern = re.compile(r'^<<<<<<< HEAD\n(.*?)\n=======\n(.*?)\n^>>>>>>> .*?$', re.DOTALL | re.MULTILINE)
    
    if not conflict_pattern.search(content):
        return {"filepath": filepath, "status": "no_conflict", "conflict_types": []}

    conflict_results = []

    def replacer(match):
        ours, theirs = match.groups()
        logging.info(f"  -> 正在解决 {filepath} 中的一个冲突...")
        result_dict = resolve_conflict_chunk(ours, theirs)
        conflict_results.append(result_dict['conflict_type'])
        logging.info(f"  -> {filepath} 中的一个冲突已解决，类型: {result_dict['conflict_type']}")
        return result_dict['merged_code']

    new_content = conflict_pattern.sub(replacer, content)

    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        return {"filepath": filepath, "status": "success", "conflict_types": conflict_results}
    except Exception as e:
        logging.error(f"无法写入文件 {filepath}: {e}")
        return {"filepath": filepath, "status": "write_error", "conflict_types": conflict_results}

def write_report(all_results):
    """将所有处理结果汇总并写入一个TXT报告文件。"""
    report_filename = f"merge_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    summary = {"total_files": len(all_results), "files_with_conflicts": 0}
    conflict_counts = {"identical": 0, "one-sided": 0, "complex": 0, "failed": 0}

    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write("AI Merge Conflict Resolution Report\n")
        f.write("="*40 + "\n")
        
        detailed_reports = []
        for result in all_results:
            filepath = result['filepath']
            types = result['conflict_types']
            
            if types:
                summary["files_with_conflicts"] += 1
                report_line = f"File: {filepath}\n"
                type_counts_per_file = {t: types.count(t) for t in set(types)}
                for t, c in type_counts_per_file.items():
                    report_line += f"  - Resolved {c} conflict(s) of type: {t}\n"
                    conflict_counts[t] += c
                detailed_reports.append(report_line)

        # 写入摘要
        f.write("Overall Summary:\n")
        f.write(f"  - Total Files Scanned: {summary['total_files']}\n")
        f.write(f"  - Files with Conflicts Processed: {summary['files_with_conflicts']}\n")
        f.write("  --- Breakdown of Resolved Conflict Types ---\n")
        f.write(f"  - Identical: {conflict_counts['identical']}\n")
        f.write(f"  - One-sided: {conflict_counts['one-sided']}\n")
        f.write(f"  - Complex: {conflict_counts['complex']}\n")
        f.write(f"  - Failed to Resolve: {conflict_counts['failed']}\n")
        f.write("="*40 + "\n\n")
        
        # 写入详细报告
        f.write("Detailed File Report:\n")
        f.write("="*40 + "\n")
        f.write("\n".join(detailed_reports))
        
    print(f"\n所有文件处理完毕。一份详细的报告已保存至: '{report_filename}'")


def main():
    """主函数，组织执行整个流程。"""
    conflicting_files = get_conflicting_files()

    if not conflicting_files:
        print("未找到需要处理的冲突文件。")
        return

    print("\n将要使用AI自动合并以下文件中的冲突：")
    for f in conflicting_files:
        print(f"  - {f}")
    
    # 移除交互式确认，根据用户指令直接执行
    print("\n根据用户指令，自动开始处理...")

    all_results = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_file = {executor.submit(process_file, f): f for f in conflicting_files}
        for future in as_completed(future_to_file):
            try:
                result = future.result()
                all_results.append(result)
            except Exception as exc:
                file_path = future_to_file[future]
                logging.error(f'文件 {file_path} 在处理过程中产生异常: {exc}')
                all_results.append({"filepath": file_path, "status": "execution_error", "conflict_types": []})

    write_report(all_results)
    print("请仔细检查被修改过的文件和报告，确认无误后手动执行 'git add .'")

if __name__ == "__main__":
    main() 