from agents import Agent, ComputerTool # 导入Agent和工具基类
import os, json, sys, asyncio

class CodeComputer:
    """代码编辑与运行工具"""
    async def edit_code(self, file, code, mode='w'): # 编辑/保存/追加代码
        try:
            with open(file, mode, encoding='utf-8') as f:
                f.write(code)
            return "代码已保存" if mode == 'w' else "代码已追加"
        except Exception as e:
            return {"error": str(e)}
    async def read_code(self, file): # 读取代码内容
        try:
            with open(file, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            return {"error": str(e)}
    async def run_code(self, file, timeout=10): # 运行Python代码，支持超时
        import subprocess
        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable, file,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            except asyncio.TimeoutError:
                proc.kill()
                return {"error": "运行超时"}
            return {
                "exit_code": proc.returncode,
                "stdout": stdout.decode('utf-8'),
                "stderr": stderr.decode('utf-8')
            }
        except Exception as e:
            return {"error": str(e)}
    async def run_shell(self, cmd, timeout=10): # 运行shell命令
        import subprocess
        try:
            proc = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            except asyncio.TimeoutError:
                proc.kill()
                return {"error": "运行超时"}
            return {
                "exit_code": proc.returncode,
                "stdout": stdout.decode('utf-8'),
                "stderr": stderr.decode('utf-8')
            }
        except Exception as e:
            return {"error": str(e)}

class CoderAgent(Agent):
    """代码编辑Agent"""
    def __init__(self):
        super().__init__(
            name="Coder Agent", # 代码编辑Agent
            instructions="代码编辑与运行智能体", # 角色描述
            tools=[ComputerTool(CodeComputer())], # 注入工具
            model="computer-use-preview" # 使用统一模型
        )
        import sys; sys.stderr.write('✅ CoderAgent初始化完成\n')
    async def handle_handoff(self, data: dict) -> str:
        try:
            action = data.get("action")
            file = data.get("file")
            code = data.get("code")
            mode = data.get("mode", "w")
            tool = CodeComputer()  # 直接新建实例
            if action == "edit":
                result = await tool.edit_code(file, code, mode)
                return json.dumps({"status": "ok", "message": "编辑成功", "data": {"result": result}}, ensure_ascii=False)
            elif action == "read":
                result = await tool.read_code(file)
                return json.dumps({"status": "ok", "message": "读取成功", "data": {"result": result}}, ensure_ascii=False)
            elif action == "run":
                result = await tool.run_code(file)
                return json.dumps({"status": "ok", "message": "运行成功", "data": {"result": result}}, ensure_ascii=False)
            elif action == "shell":
                cmd = data.get("cmd")
                result = await tool.run_shell(cmd)
                return json.dumps({"status": "ok", "message": "Shell命令执行成功", "data": {"result": result}}, ensure_ascii=False)
            else:
                return json.dumps({"status": "error", "message": "未知操作", "data": {}}, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"status": "error", "message": str(e), "data": {}}, ensure_ascii=False)

# 工厂函数：动态创建Agent实例
def create_coder_agent():
    """创建CoderAgent实例"""
    return CoderAgent()

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
