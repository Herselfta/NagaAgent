# Playwright 插件升级说明 #

## 三层Agent架构设计 #

### 架构概览 #
- **ControllerAgent**（顶层调度）：负责理解用户目标，智能分配任务，汇总结果 #
- **BrowserAgent**（页面操作）：负责网页自动化操作，如打开、点击、输入、滚动、截图等 #
- **ContentAgent**（内容处理）：负责网页内容清洗、摘要、翻译等处理 #

### 协作流程 #
```
用户请求 → ControllerAgent → 智能分配 → BrowserAgent/ContentAgent → 结果汇总
```

## 核心组件说明 #

### 1. ControllerAgent（agent_controller.py） #
- **职责**：顶层调度，理解用户意图，分配任务给合适的Agent #
- **工具**：无直接工具，通过handoff调用其他Agent #
- **handoffs**：BrowserAgent、ContentAgent #
- **生命周期钩子**：ControllerAgentHooks #

### 2. BrowserAgent（controller.py） #
- **职责**：网页自动化操作 #
- **工具**：PlaywrightController #
- **支持操作**：
  - open_url：打开指定URL #
  - click：点击指定元素 #
  - type：在指定元素输入文本 #
  - scroll：滚动页面 #
  - wait_for_element：等待元素出现 #
  - take_screenshot：截图 #
  - search_github：GitHub搜索自动化 #
- **生命周期钩子**：BrowserAgentHooks #

### 3. ContentAgent（browser.py） #
- **职责**：内容处理、清洗、摘要、翻译 #
- **工具**：PlaywrightBrowser #
- **支持操作**：
  - get_content：获取页面内容（支持Markdown/HTML格式，自动去广告） #
  - get_title：获取页面标题 #
  - get_screenshot：获取页面截图 #
  - subscribe_page_change：推送式订阅页面内容变化 #

## 底层工具类 #

### PlaywrightController #
- 页面操作控制器，负责自动化交互 #
- 支持点击、输入、滚动、等待、截图等操作 #

### PlaywrightBrowser #
- 浏览器观察器，负责内容获取和推送 #
- 支持内容清洗、格式转换、变化监听 #

## 调用示例 #

### 1. MCP Handoff调用 #
```text
用户：打开百度并搜索"Python教程"
流程：MCP handoff → ControllerAgent → BrowserAgent → PlaywrightController
```

### 2. 页面操作（通过ControllerAgent → BrowserAgent） #
```text
用户：打开百度并搜索"Python教程"
流程：ControllerAgent → BrowserAgent → PlaywrightController
```

### 3. 内容处理（通过ControllerAgent → ContentAgent） #
```text
用户：获取网页内容并翻译成中文
流程：ControllerAgent → ContentAgent → PlaywrightBrowser
```

### 4. 复杂任务（多Agent协作） #
```text
用户：打开GitHub搜索Python项目，获取README并翻译
流程：ControllerAgent → BrowserAgent（搜索） → ContentAgent（翻译）
```

### 5. MCP Handoff参数示例 #
```json
{
  "action": "open",
  "url": "https://www.baidu.com",
  "task_type": "browser"
}
```

```json
{
  "action": "get_content",
  "format": "markdown",
  "task_type": "content"
}
```

## 推送式页面观察用法 #

```python
from agent_playwright_master.browser import PlaywrightBrowser

async def on_content_change(content):
    print('页面内容变化:', content)

browser = PlaywrightBrowser(page)
await browser.subscribe_page_change(on_content_change, interval=2.0)  # 每2秒推送一次 #
# ...如需停止推送：browser.stop_subscribe() #
```

## 生命周期监控 #

所有Agent都集成了生命周期钩子，提供执行日志：
- `on_start`：Agent开始执行时触发 #
- `on_end`：Agent结束执行时触发 #

## 环境变量配置 #

```bash
DEEPSEEK_API_KEY=your_api_key
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
```

## 文件结构 #

```
agent_playwright_master/
├── agent_controller.py    # ControllerAgent（顶层调度）
├── controller.py          # BrowserAgent（页面操作）
├── browser.py             # ContentAgent（内容处理）
├── __init__.py           # 模块导出
└── README.md             # 本文档
```

---

详细接口和参数请参考代码注释与 config.py 配置 # 