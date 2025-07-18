from agents import Agent, ComputerTool
import os, json, shutil
import stat as statmod  # 仅在此方法内导入，防止全局污染
import asyncio

class FileComputer:
    """文件操作工具（增强版）"""
    DESKTOP_DIR = os.path.join(os.path.expanduser("~"), "Desktop")  # 仅支持Windows桌面
    BASE_DIR = DESKTOP_DIR  # 默认根目录为桌面

    def _safe_path(self, path): # 路径安全校验，防止目录穿越
        # 支持~/Desktop/xxx 或 desktop/xxx 自动映射到桌面
        norm_path = path.replace("\\", "/")
        if norm_path.startswith("~/Desktop/"):
            abs_path = os.path.abspath(os.path.join(self.DESKTOP_DIR, norm_path[10:]))
        elif norm_path.lower().startswith("desktop/"):
            abs_path = os.path.abspath(os.path.join(self.DESKTOP_DIR, norm_path[8:]))
        else:
            abs_path = os.path.abspath(os.path.join(self.BASE_DIR, path))
        # 只允许BASE_DIR和DESKTOP_DIR下写入
        if abs_path.startswith(self.BASE_DIR) or abs_path.startswith(self.DESKTOP_DIR):
            return abs_path
        raise ValueError("非法路径")

    async def list_dir(self, path, limit=100): # 列目录
        try:
            abs_path = self._safe_path(path)
            files = os.listdir(abs_path)[:limit]
            return {"status": "ok", "data": files}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def read_file(self, path): # 读文件
        try:
            abs_path = self._safe_path(path)
            with open(abs_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return {"status": "ok", "data": content}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def write_file(self, path, content, append=False): # 写/追加文件
        try:
            abs_path = self._safe_path(path)
            mode = 'a' if append else 'w'
            with open(abs_path, mode, encoding='utf-8') as f:
                f.write(content)
            return {"status": "ok", "message": "写入成功" if not append else "追加成功"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def delete_file(self, path, recursive=False): # 删除文件/目录
        try:
            abs_path = self._safe_path(path)
            if os.path.isdir(abs_path):
                if recursive:
                    shutil.rmtree(abs_path)
                    return {"status": "ok", "message": "目录递归删除成功"}
                else:
                    os.rmdir(abs_path)
                    return {"status": "ok", "message": "空目录删除成功"}
            else:
                os.remove(abs_path)
                return {"status": "ok", "message": "文件删除成功"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def make_dir(self, path): # 创建目录
        try:
            abs_path = self._safe_path(path)
            os.makedirs(abs_path, exist_ok=True)
            return {"status": "ok", "message": "目录创建成功"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def create_file(self, path): # 创建空文件
        try:
            abs_path = self._safe_path(path)
            with open(abs_path, 'w', encoding='utf-8') as f:
                pass
            return {"status": "ok", "message": "空文件创建成功"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def move(self, src, dst): # 移动文件/目录
        try:
            abs_src = self._safe_path(src)
            abs_dst = self._safe_path(dst)
            shutil.move(abs_src, abs_dst)
            return {"status": "ok", "message": "移动成功"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def copy(self, src, dst): # 复制文件/目录
        try:
            abs_src = self._safe_path(src)
            abs_dst = self._safe_path(dst)
            if os.path.isdir(abs_src):
                shutil.copytree(abs_src, abs_dst)
            else:
                shutil.copy2(abs_src, abs_dst)
            return {"status": "ok", "message": "复制成功"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def rename(self, src, dst): # 重命名文件/目录
        try:
            abs_src = self._safe_path(src)
            abs_dst = self._safe_path(dst)
            os.rename(abs_src, abs_dst)
            return {"status": "ok", "message": "重命名成功"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def stat(self, path): # 获取文件/目录详细属性
        try:
            abs_path = self._safe_path(path)
            s = os.stat(abs_path)
            info = {
                "size": s.st_size, # 文件大小
                "mtime": s.st_mtime, # 修改时间
                "ctime": s.st_ctime, # 创建时间
                "is_dir": os.path.isdir(abs_path), # 是否为目录
                "is_file": os.path.isfile(abs_path), # 是否为文件
                "mode": s.st_mode, # 权限mode
                "permissions": statmod.filemode(s.st_mode), # 权限字符串
                "uid": getattr(s, 'st_uid', None), # 所有者UID
                "gid": getattr(s, 'st_gid', None), # 所有者GID
                "inode": getattr(s, 'st_ino', None), # inode编号
                "nlink": getattr(s, 'st_nlink', None), # 硬链接数
            }
            return {"status": "ok", "data": info}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def batch_delete(self, paths, recursive=False): # 批量删除文件/目录
        results = [] # 结果列表
        for path in paths:
            try:
                abs_path = self._safe_path(path)
                if os.path.isdir(abs_path):
                    if recursive:
                        shutil.rmtree(abs_path)
                        results.append({"path": path, "status": "ok", "message": "目录递归删除成功"})
                    else:
                        os.rmdir(abs_path)
                        results.append({"path": path, "status": "ok", "message": "空目录删除成功"})
                else:
                    os.remove(abs_path)
                    results.append({"path": path, "status": "ok", "message": "文件删除成功"})
            except Exception as e:
                results.append({"path": path, "status": "error", "message": str(e)})
        return {"status": "ok", "results": results}

    async def read_file_stream(self, path, offset=0, size=4096): # 流式读取大文件
        """按块读取大文件，offset为起始字节，size为读取字节数"""
        try:
            abs_path = self._safe_path(path)
            with open(abs_path, 'rb') as f:
                f.seek(offset)
                data = f.read(size)
            return {"status": "ok", "data": data}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def write_file_stream(self, path, data, offset=0): # 流式写入大文件
        """按块写入大文件，offset为写入起始字节，data为bytes"""
        try:
            abs_path = self._safe_path(path)
            with open(abs_path, 'r+b') as f:
                f.seek(offset)
                f.write(data)
            return {"status": "ok", "message": "流式写入成功"}
        except FileNotFoundError:
            # 文件不存在则新建
            try:
                with open(abs_path, 'wb') as f:
                    f.seek(offset)
                    f.write(data)
                return {"status": "ok", "message": "新建并流式写入成功"}
            except Exception as e:
                return {"status": "error", "message": str(e)}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def search_files(self, path, pattern=None, recursive=True, filter_func=None, limit=100): # 文件搜索与过滤
        """在指定目录下搜索文件，支持通配符、递归和自定义过滤"""
        import fnmatch
        result = []
        try:
            abs_path = self._safe_path(path)
            for root, dirs, files in os.walk(abs_path):
                for name in files:
                    if pattern is None or fnmatch.fnmatch(name, pattern):
                        file_path = os.path.join(root, name)
                        if filter_func is None or filter_func(file_path):
                            result.append(file_path)
                            if len(result) >= limit:
                                return {"status": "ok", "data": result}
                if not recursive:
                    break
            return {"status": "ok", "data": result}
        except Exception as e:
            return {"status": "error", "message": str(e)}

class FileAgent(Agent):
    """文件操作Agent"""
    def __init__(self):
        super().__init__(
            name="File Agent", # 文件操作Agent
            instructions="文件操作智能体", # 角色描述
            tools=[ComputerTool(FileComputer())], # 注入工具
            model="file-use-preview" # 使用统一模型
        )
        import sys; sys.stderr.write('✅ FileAgent初始化完成\n')

    async def handle_handoff(self, data: dict) -> str:
        try:
            action = data.get("action")
            path = data.get("path")
            tool = FileComputer()  # 直接新建实例，
            if action == "read":
                result = await tool.read_file(path)
                if result["status"] == "ok":
                    content = result["data"]
                    preview = content[:500] + ("...（内容过长仅展示前500字）" if len(content) > 500 else "")
                    return json.dumps({"status": "ok", "message": "读取成功", "data": {"content": preview}}, ensure_ascii=False)
                else:
                    return json.dumps({"status": "error", "message": result.get("message", "读取失败"), "data": {}}, ensure_ascii=False)
            elif action == "list":
                result = await tool.list_dir(path)
                if result["status"] == "ok":
                    return json.dumps({"status": "ok", "message": "列目录成功", "data": {"files": result["data"]}}, ensure_ascii=False)
                else:
                    return json.dumps({"status": "error", "message": result.get("message", "列目录失败"), "data": {}}, ensure_ascii=False)
            else:
                return json.dumps({"status": "error", "message": "未知操作", "data": {}}, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"status": "error", "message": str(e), "data": {}}, ensure_ascii=False)

# 工厂函数：动态创建Agent实例
def create_file_agent():
    """创建FileAgent实例"""
    return FileAgent()

# 获取Agent元数据
def get_agent_metadata():
    """获取Agent元数据"""
    import os
    manifest_path = os.path.join(os.path.dirname(__file__), "agent-manifest.json")
    try:
        with open(manifest_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"加载元数据失败: {e}")
        return None

# 验证配置
def validate_agent_config(config):
    """验证Agent配置"""
    return True

# 获取依赖
def get_agent_dependencies():
    """获取Agent依赖"""
    return []