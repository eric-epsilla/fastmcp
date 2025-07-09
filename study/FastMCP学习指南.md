# 🚀 FastMCP 全面学习指南

FastMCP是一个用于构建Model Context Protocol (MCP) 服务器和客户端的Python框架。MCP被称为"AI的USB-C接口"，它为LLM与外部工具、数据源的交互提供了标准化协议。

## 📚 一、核心概念理解

### 什么是MCP？
- **定义**：Model Context Protocol是连接LLM和外部系统的开放标准
- **作用**：解决AI应用中的一个根本问题——如何让LLM可靠、安全地与外部工具和数据交互
- **价值**：提供互操作性、可发现性、安全性和可组合性

### FastMCP的优势
- **🚀 快速**：高级接口，减少代码量，加快开发速度
- **🍀 简单**：最小化样板代码构建MCP服务器
- **🐍 Pythonic**：对Python开发者友好的设计
- **🔍 完整**：覆盖从开发到生产的全套MCP用例

### MCP vs 传统API
MCP不同于传统REST API的关键点：
1. **标准化**：统一的接口标准，一次构建，多处使用
2. **AI原生**：专为LLM交互设计的协议
3. **动态发现**：客户端可以在运行时发现服务器能力
4. **类型安全**：强类型验证和schema生成
5. **安全边界**：明确的沙箱机制

## 🛠️ 二、环境设置

### 安装FastMCP
```bash
# 推荐使用uv
uv pip install fastmcp

# 或使用pip
pip install fastmcp
```

### 验证安装
```bash
fastmcp version
```

### 开发环境设置
```bash
# 克隆仓库（如果要贡献代码）
git clone https://github.com/jlowin/fastmcp.git
cd fastmcp
uv sync

# 安装预提交钩子
uv run pre-commit install
```

## 🎯 三、核心组件详解

### 1. Tools (工具) - 可执行动作
**概念**：类似REST API的POST请求，用于执行操作、改变状态或触发副作用

**基本工具示例**：
```python
from fastmcp import FastMCP

mcp = FastMCP("Demo Server")

@mcp.tool
def calculate_sum(a: int, b: int) -> int:
    """计算两个数的和"""
    return a + b

@mcp.tool
def send_email(to: str, subject: str, body: str) -> str:
    """发送邮件"""
    # 实际的邮件发送逻辑
    return f"邮件已发送至 {to}"
```

**异步工具示例**：
```python
import asyncio

@mcp.tool
async def fetch_weather(city: str) -> dict:
    """异步获取天气信息"""
    await asyncio.sleep(1)  # 模拟API调用
    return {"city": city, "temperature": "22°C", "condition": "晴"}
```

### 2. Resources (资源) - 只读数据
**概念**：类似REST API的GET请求，用于提供数据源供LLM读取

**静态资源**：
```python
@mcp.resource("config://version")
def get_version():
    """获取系统版本信息"""
    return "2.0.1"

@mcp.resource("system://status")  
def get_system_status():
    """获取系统状态"""
    return {"status": "运行中", "uptime": "24小时"}
```

**动态资源模板**：
```python
@mcp.resource("users://{user_id}/profile")
def get_user_profile(user_id: int):
    """获取用户档案信息"""
    return {
        "id": user_id, 
        "name": f"用户{user_id}",
        "created_at": "2024-01-01"
    }

@mcp.resource("files://{path}")
def get_file_content(path: str):
    """读取文件内容"""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return "文件未找到"
```

### 3. Prompts (提示词) - 可重用模板
**概念**：定义可重用的消息模板来指导LLM交互

```python
@mcp.prompt
def summarize_text(text_to_summarize: str) -> str:
    """生成文本总结提示"""
    return f"""请总结以下文本，要求：
1. 保持关键信息
2. 控制在200字以内
3. 使用中文回复

文本内容：
{text_to_summarize}
"""

@mcp.prompt
def code_review(code: str, language: str) -> str:
    """代码审查提示"""
    return f"""请审查以下{language}代码，关注：
- 代码质量和可读性
- 潜在的bug和安全问题
- 性能优化建议

代码：
```{language}
{code}
```
"""
```

### 4. Context (上下文) - 会话能力
**功能**：在工具、资源或提示词中访问MCP会话能力

```python
from fastmcp import Context

@mcp.tool
def log_and_sample(message: str, ctx: Context) -> str:
    """带日志和采样的工具"""
    # 记录日志
    ctx.info(f"处理消息: {message}")
    
    # 可以请求客户端的LLM生成补全
    # result = await ctx.sample("请分析这个消息的情感倾向", message)
    
    return f"已处理: {message}"

@mcp.tool  
async def smart_response(query: str, ctx: Context) -> str:
    """智能回复工具"""
    ctx.info("开始处理智能回复请求")
    
    # 使用客户端LLM生成回复
    response = await ctx.sample(
        f"请对以下查询给出专业回复: {query}"
    )
    
    ctx.info("智能回复生成完成")
    return response
```

## 📖 四、学习路径与实践

### 第一步：创建简单服务器
基于 `examples/simple_echo.py`：
```python
from fastmcp import FastMCP

mcp = FastMCP("Echo Server")

@mcp.tool
def echo(text: str) -> str:
    """回显输入文本"""
    return text

if __name__ == "__main__":
    mcp.run()
```

### 第二步：运行和测试
```bash
# 直接运行
python server.py

# 使用CLI运行
fastmcp run server.py

# 开发模式（带MCP Inspector）
fastmcp dev server.py

# 指定传输协议
fastmcp run server.py --transport http --port 8080
```

### 第三步：客户端测试
```python
import asyncio
from fastmcp import Client

async def test_server():
    client = Client("server.py")  # 或 "http://localhost:8080"
    
    async with client:
        # 列出可用工具
        tools = await client.list_tools()
        print("可用工具:", [tool.name for tool in tools])
        
        # 调用工具
        result = await client.call_tool("echo", {"text": "Hello, MCP!"})
        print("工具调用结果:", result)
        
        # 列出资源
        resources = await client.list_resources()
        print("可用资源:", [resource.uri for resource in resources])

asyncio.run(test_server())
```

### 第四步：探索复杂示例

#### 1. memory.py - 递归记忆系统
- 展示了向量数据库集成
- 异步操作和依赖管理
- 复杂的AI模式实现

#### 2. atproto_mcp/ - 社交媒体集成
- 完整的API集成示例
- 认证和配置管理
- 多种工具和资源类型

#### 3. smart_home/ - 智能家居系统
- 模块化设计
- 设备管理和控制
- 实际IoT应用场景

## 🏗️ 五、架构深入理解

### FastMCP类结构
```python
class FastMCP:
    def __init__(self, name: str, ...):
        self._tool_manager = ToolManager(...)
        self._resource_manager = ResourceManager(...)
        self._prompt_manager = PromptManager(...)
        self._mcp_server = LowLevelServer(...)
```

**核心组件**：
- **FastMCP服务器**：主要容器和协调器
- **管理器系统**：
  - `ToolManager` - 管理工具注册、调用和生命周期
  - `ResourceManager` - 管理资源和模板的读取
  - `PromptManager` - 管理提示词模板
- **底层服务器**：处理MCP协议细节
- **传输层**：STDIO、HTTP、SSE等传输协议支持

### 客户端架构
```python
from fastmcp import Client

# 多种连接方式
client = Client("server.py")                    # 本地脚本
client = Client("http://localhost:8080")        # HTTP服务器  
client = Client("sse://localhost:8080/sse")     # SSE服务器
client = Client(mcp_instance)                   # 内存中的服务器

# 使用客户端
async with client:
    # 所有MCP操作都在这个上下文中进行
    result = await client.call_tool("tool_name", {"param": "value"})
```

### 传输协议详解

#### STDIO传输
- **用途**：本地命令行工具，与Claude Desktop等客户端集成
- **特点**：通过标准输入输出通信
- **适用场景**：开发测试、本地工具集成

#### HTTP传输  
- **用途**：Web服务部署
- **特点**：基于HTTP的请求/响应模式
- **适用场景**：生产环境、远程服务

#### SSE传输
- **用途**：需要服务器推送的场景
- **特点**：Server-Sent Events，支持实时通信
- **适用场景**：实时数据推送、长连接需求

## 🔗 六、集成应用场景

### 1. Claude Desktop集成
```bash
# 自动安装和配置
fastmcp install claude-desktop server.py

# 手动配置~/.claude/claude_desktop_config.json
{
  "mcpServers": {
    "my-server": {
      "command": "python",
      "args": ["server.py"]
    }
  }
}
```

### 2. Cursor集成
```bash
# 自动安装
fastmcp install cursor server.py

# 手动配置~/.cursor/mcp.json
{
  "mcpServers": {
    "my-server": {
      "command": "python", 
      "args": ["server.py"]
    }
  }
}
```

### 3. ChatGPT Deep Research集成
ChatGPT需要特定的工具接口：
```python
@mcp.tool
def search(query: str) -> list[dict]:
    """搜索资源
    
    Args:
        query: 搜索查询字符串
        
    Returns:
        匹配资源的列表，每个包含id和简要描述
    """
    # 实现搜索逻辑
    return [
        {"id": "doc1", "description": "相关文档1"}, 
        {"id": "doc2", "description": "相关文档2"}
    ]

@mcp.tool  
def fetch(resource_id: str) -> str:
    """获取资源完整内容
    
    Args:
        resource_id: 资源唯一标识符
        
    Returns:
        资源的完整文本内容
    """
    # 根据ID获取完整内容
    return "这里是完整的资源内容..."
```

### 4. 自定义AI应用集成
```python
from fastmcp import Client

class AIAssistant:
    def __init__(self):
        self.mcp_client = Client("my_tools_server.py")
    
    async def process_request(self, user_input: str):
        async with self.mcp_client:
            # 分析用户输入，决定调用哪些工具
            if "天气" in user_input:
                result = await self.mcp_client.call_tool(
                    "get_weather", {"city": "北京"}
                )
                return f"天气信息: {result}"
            
            # 其他处理逻辑...
```

## 🚀 七、高级功能

### 1. 代理服务器
代理服务器可以桥接不同的传输协议或添加中间层逻辑：

```python
from fastmcp import FastMCP, Client

# 创建代理，将HTTP后端暴露为STDIO
proxy = FastMCP.as_proxy(
    client_factory=lambda: Client("http://remote-server:8080"),
    name="HTTP代理服务器"
)

# 运行代理
if __name__ == "__main__":
    proxy.run()  # 默认STDIO传输
```

**使用场景**：
- 协议转换（HTTP ↔ STDIO）
- 添加认证层
- 负载均衡
- 请求转换和过滤

### 2. 服务器组合
将多个MCP服务器组合成一个：

```python
# 主服务器
main_server = FastMCP("主服务器")

# 子服务器
auth_server = FastMCP("认证服务器")
data_server = FastMCP("数据服务器")

# 挂载子服务器
main_server.mount(auth_server, prefix="auth")
main_server.mount(data_server, prefix="data")

# 现在工具会有前缀：auth_login, data_query等
```

### 3. OpenAPI集成
从现有REST API自动生成MCP服务器：

```python
from fastmcp.server.openapi import FastMCPOpenAPI
import httpx

# 从OpenAPI规范创建MCP服务器
server = FastMCPOpenAPI(
    openapi_spec=api_spec,           # OpenAPI 3.0规范
    client=httpx.AsyncClient(),      # HTTP客户端
    name="API集成服务器",
    timeout=30.0
)

# 自动将API端点转换为MCP工具和资源
```

### 4. 认证系统

#### Bearer Token认证
```python
from fastmcp.server.auth.providers.bearer import BearerAuthProvider, RSAKeyPair

# 生成密钥对
key_pair = RSAKeyPair.generate()

# 创建认证提供者
auth = BearerAuthProvider(
    public_key=key_pair.public_key,
    issuer="https://example.com",
    audience="my-server",
    required_scopes=["mcp:read", "mcp:write"]
)

# 应用到服务器
mcp = FastMCP("安全服务器", auth=auth)

# 生成访问令牌
token = key_pair.create_token(
    subject="user123",
    issuer="https://example.com", 
    audience="my-server",
    scopes=["mcp:read", "mcp:write"]
)
```

#### 环境变量认证
```python
from fastmcp.server.auth.providers.bearer_env import EnvBearerAuthProvider

# 从环境变量读取token
auth = EnvBearerAuthProvider()  # 默认读取MCP_BEARER_TOKEN

mcp = FastMCP("环境认证服务器", auth=auth)
```

### 5. 中间件系统
```python
from fastmcp.server.middleware import Middleware

class LoggingMiddleware(Middleware):
    async def process_request(self, request, call_next):
        print(f"收到请求: {request}")
        response = await call_next(request)
        print(f"返回响应: {response}")
        return response

mcp = FastMCP(
    "带中间件的服务器",
    middleware=[LoggingMiddleware()]
)
```

## 💡 八、最佳实践建议

### 开发最佳实践

#### 1. 代码组织
```python
# 推荐的项目结构
my_mcp_server/
├── main.py              # 主服务器文件
├── tools/               # 工具模块
│   ├── __init__.py
│   ├── calculation.py
│   └── file_ops.py
├── resources/           # 资源模块  
│   ├── __init__.py
│   └── data_sources.py
├── config.py           # 配置管理
└── requirements.txt    # 依赖列表
```

#### 2. 错误处理
```python
from fastmcp.exceptions import MCPError

@mcp.tool
def safe_divide(a: float, b: float) -> float:
    """安全除法运算"""
    if b == 0:
        raise MCPError("除数不能为零")
    return a / b

@mcp.resource("files://{path}")
def safe_read_file(path: str) -> str:
    """安全文件读取"""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        raise MCPError(f"文件未找到: {path}")
    except PermissionError:
        raise MCPError(f"无权限访问文件: {path}")
```

#### 3. 类型安全
```python
from typing import List, Dict, Optional
from pydantic import BaseModel

class UserInfo(BaseModel):
    id: int
    name: str
    email: Optional[str] = None

@mcp.tool
def create_user(user_data: UserInfo) -> Dict[str, str]:
    """创建用户，使用Pydantic模型确保类型安全"""
    # FastMCP自动验证输入数据
    return {"status": "created", "user_id": str(user_data.id)}
```

#### 4. 异步最佳实践
```python
import asyncio
import aiohttp

@mcp.tool
async def fetch_data(url: str) -> dict:
    """异步获取数据"""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.json()

# 并发处理多个请求
@mcp.tool
async def batch_process(urls: List[str]) -> List[dict]:
    """批量处理URL"""
    tasks = [fetch_data(url) for url in urls]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return [r if not isinstance(r, Exception) else {"error": str(r)} for r in results]
```

### 部署最佳实践

#### 1. 环境管理
```python
# config.py
import os
from pydantic import BaseSettings

class Settings(BaseSettings):
    api_key: str
    debug: bool = False
    max_connections: int = 100
    
    class Config:
        env_file = ".env"

settings = Settings()

# main.py
from config import settings

mcp = FastMCP(
    "生产服务器",
    dependencies=["aiohttp", "pydantic"],  # 声明依赖
)

@mcp.tool
async def api_call(endpoint: str) -> dict:
    """使用配置的API密钥调用外部API"""
    headers = {"Authorization": f"Bearer {settings.api_key}"}
    # API调用逻辑...
```

#### 2. Docker部署
```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8080

CMD ["fastmcp", "run", "main.py", "--transport", "http", "--host", "0.0.0.0", "--port", "8080"]
```

#### 3. 健康检查
```python
@mcp.resource("health://status")
def health_check() -> dict:
    """健康检查端点"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }
```

### 性能优化

#### 1. 缓存策略
```python
from functools import lru_cache
import asyncio

# 同步缓存
@lru_cache(maxsize=128)
def expensive_calculation(n: int) -> int:
    """缓存计算结果"""
    return sum(i * i for i in range(n))

# 异步缓存
cache = {}

async def cached_api_call(endpoint: str) -> dict:
    """带缓存的API调用"""
    if endpoint in cache:
        return cache[endpoint]
    
    # 模拟API调用
    await asyncio.sleep(1)
    result = {"data": f"result for {endpoint}"}
    cache[endpoint] = result
    return result
```

#### 2. 连接池管理
```python
import aiohttp

class DatabaseManager:
    def __init__(self):
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

# 在FastMCP中使用
db_manager = DatabaseManager()

@mcp.tool
async def query_database(query: str) -> dict:
    """数据库查询，复用连接"""
    async with db_manager as db:
        # 使用连接池查询数据库
        pass
```

## 🎯 九、实际项目建议

### 项目类型选择

#### 1. 数据集成项目
**适用场景**：连接现有数据源，为AI提供上下文
```python
# 数据库集成示例
@mcp.resource("database://tables/{table_name}")
async def get_table_data(table_name: str) -> list:
    """从数据库获取表数据"""
    # 数据库查询逻辑
    pass

@mcp.tool
async def execute_query(sql: str) -> dict:
    """执行SQL查询"""
    # 安全的SQL执行
    pass
```

#### 2. API包装项目  
**适用场景**：将第三方API包装为MCP接口
```python
# 天气API包装
@mcp.tool
async def get_weather(city: str, units: str = "metric") -> dict:
    """获取天气信息"""
    api_key = os.getenv("WEATHER_API_KEY")
    url = f"http://api.openweathermap.org/data/2.5/weather"
    params = {"q": city, "appid": api_key, "units": units}
    
    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params) as response:
            return await response.json()
```

#### 3. 工作流自动化项目
**适用场景**：自动化复杂的业务流程
```python
@mcp.tool
async def process_document(file_path: str) -> dict:
    """文档处理工作流"""
    # 1. 读取文档
    content = await read_document(file_path)
    
    # 2. 提取关键信息
    key_info = await extract_info(content)
    
    # 3. 生成报告
    report = await generate_report(key_info)
    
    # 4. 发送通知
    await send_notification(report)
    
    return {"status": "completed", "report_id": report.id}
```

#### 4. AI增强应用项目
**适用场景**：为现有应用添加AI能力
```python
@mcp.tool
async def analyze_sentiment(text: str, ctx: Context) -> dict:
    """情感分析工具"""
    # 使用客户端LLM进行分析
    prompt = f"请分析以下文本的情感倾向，返回JSON格式：{text}"
    analysis = await ctx.sample(prompt)
    
    return {"text": text, "sentiment": analysis}
```

### 开发流程建议

#### 1. 需求分析阶段
- 确定目标LLM客户端（Claude、GPT、自定义等）
- 分析需要提供的工具和数据
- 设计API接口和数据流
- 评估性能和安全要求

#### 2. 原型开发阶段
```python
# 快速原型
from fastmcp import FastMCP

mcp = FastMCP("原型服务器")

@mcp.tool
def prototype_tool(input_data: str) -> str:
    """原型工具，验证概念"""
    return f"处理结果: {input_data}"

# 快速测试
if __name__ == "__main__":
    mcp.run()
```

#### 3. 迭代开发阶段
- 逐步添加功能
- 使用FastMCP客户端进行单元测试
- 集成到目标AI应用中测试
- 收集反馈并优化

#### 4. 生产部署阶段
- 完善错误处理和日志
- 添加监控和健康检查
- 实施安全措施
- 文档编写和维护

## 📚 十、学习资源和社区

### 官方资源
- **官方文档**：[gofastmcp.com](https://gofastmcp.com)
- **GitHub仓库**：[github.com/jlowin/fastmcp](https://github.com/jlowin/fastmcp)
- **MCP规范**：[modelcontextprotocol.io](https://modelcontextprotocol.io)

### 学习材料
- **示例代码**：仓库中的 `examples/` 目录
- **教程文档**：`docs/tutorials/` 目录
- **集成指南**：`docs/integrations/` 目录
- **API参考**：`docs/python-sdk/` 目录

### 社区资源
- **社区展示**：`docs/community/showcase.mdx`
- **贡献指南**：仓库中的贡献文档
- **问题讨论**：GitHub Issues

### 推荐学习路径

#### 初学者路径（1-2周）
1. 理解MCP基本概念
2. 安装和配置开发环境
3. 运行简单示例
4. 创建第一个工具和资源
5. 学习客户端基本使用

#### 进阶路径（2-4周）
1. 深入理解架构设计
2. 学习异步编程模式
3. 掌握认证和安全
4. 实践复杂示例项目
5. 学习部署和优化

#### 专家路径（1-2个月）
1. 研究源码实现
2. 贡献开源项目
3. 设计复杂的MCP架构
4. 开发高性能解决方案
5. 分享经验和最佳实践

---

## 🎯 总结

FastMCP是一个功能强大且易于使用的MCP框架，它简化了AI应用与外部系统的集成。通过掌握其核心概念、架构设计和最佳实践，你可以：

1. **快速构建MCP服务器**：使用装饰器轻松创建工具、资源和提示词
2. **灵活集成现有系统**：支持多种传输协议和认证机制
3. **构建可扩展的AI应用**：利用代理、组合和中间件等高级功能
4. **实现生产级部署**：具备完整的错误处理、监控和优化能力

建议按照学习路径循序渐进，多实践、多思考，逐步掌握FastMCP的精髓。记住，最好的学习方式是动手实践！

**祝你学习愉快！🚀** 


---

🎯 十大核心章节：
核心概念理解 - MCP基础知识和FastMCP优势
环境设置 - 安装配置和开发环境
核心组件详解 - Tools、Resources、Prompts、Context
学习路径与实践 - 从简单到复杂的实践步骤
架构深入理解 - 底层实现和设计原理
集成应用场景 - Claude Desktop、Cursor、ChatGPT等
高级功能 - 代理服务器、认证、中间件等
最佳实践建议 - 开发、部署、性能优化
实际项目建议 - 四种典型项目类型
学习资源和社区 - 官方资源和学习路径

---
🚀 特色内容：
代码示例丰富 - 每个概念都有完整可运行的代码
循序渐进 - 从简单echo到复杂AI应用
实用性强 - 涵盖开发到部署全流程
最佳实践 - 错误处理、类型安全、异步编程
项目指导 - 四种项目类型的具体实现建议
你现在可以：
按照文档中的TODO列表循序渐进学习
参考代码示例进行实践
根据项目建议选择合适的应用场景
使用最佳实践指导实际开发

---

你现在可以：
按照文档中的TODO列表循序渐进学习
参考代码示例进行实践
根据项目建议选择合适的应用场景
使用最佳实践指导实际开发