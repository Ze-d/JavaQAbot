# 🔍 医疗QA聊天机器人 - 调试日志系统使用指南

## 📋 概述

本项目已添加完整的调试日志系统，用于追踪系统运行状态、定位问题和性能优化。日志系统采用 Python 标准库 `logging`，支持多级别输出和文件持久化。

## 🗂️ 日志文件结构

```
logs/
├── medical_qa_20241215_143022.log  # 主日志文件（带时间戳）
└── ...
```

## 📊 日志级别说明

| 级别 | 颜色标识 | 用途 | 场景示例 |
|------|----------|------|----------|
| **DEBUG** | 灰色 | 详细调试信息 | 方法参数、变量值、流程步骤 |
| **INFO** | 绿色 | 重要操作记录 | 初始化成功、方法调用完成 |
| **WARNING** | 黄色 | 潜在问题 | 降级处理、参数调整 |
| **ERROR** | 红色 | 错误异常 | 异常捕获、失败记录 |

## 🎯 各模块日志详情

### 1️⃣ **logger_config.py** - 日志配置模块

#### 📍 类/函数：无
#### 🔍 日志类型：配置级别
#### 📝 说明：统一配置各模块的日志记录器

```python
# 创建的日志记录器
logger_agent = setup_logger('Agent', 'DEBUG')      # Agent模块 - DEBUG级别
logger_service = setup_logger('Service', 'INFO')   # Service模块 - INFO级别
logger_utils = setup_logger('Utils', 'INFO')       # Utils模块 - INFO级别
logger_app = setup_logger('App', 'INFO')           # App模块 - INFO级别
logger_data = setup_logger('DataProcess', 'INFO')  # 数据处理模块 - INFO级别
```

---

### 2️⃣ **utils.py** - 工具函数模块

#### 📍 **get_embeddings_model()** 方法
- **日志级别**：DEBUG / INFO / ERROR
- **作用**：追踪嵌入模型初始化过程

```python
DEBUG: "开始初始化嵌入模型"
INFO: "设备选择: cuda/cpu"
DEBUG: "模型参数: {...}"
DEBUG: "编码参数: {...}"
INFO: "模型路径: C:/02-study/model/embeding/bge-base"
INFO: "嵌入模型初始化成功"
ERROR: "嵌入模型初始化失败: {error}"
```

#### 📍 **get_llm_model()** 方法
- **日志级别**：DEBUG / INFO / ERROR
- **作用**：追踪LLM模型获取过程

```python
DEBUG: "开始获取LLM模型"
INFO: "LLM模型获取成功"
DEBUG: "模型参数: temperature=0.01, max_tokens=2048"
ERROR: "LLM模型获取失败: {error}"
```

#### 📍 **get_neo4j_conn()** 方法
- **日志级别**：DEBUG / INFO / ERROR
- **作用**：追踪Neo4j数据库连接

```python
DEBUG: "开始连接Neo4j数据库"
DEBUG: "连接参数: uri={uri}, username={username}"
INFO: "Neo4j连接成功"
ERROR: "Neo4j连接失败: {error}"
```

#### 📍 **structured_output_parser()** 方法
- **日志级别**：DEBUG
- **作用**：追踪结构化输出提示词生成

```python
DEBUG: "开始生成结构化输出提示词"
DEBUG: "响应模式数量: {count}"
DEBUG: "提示词长度: {length} 字符"
```

#### 📍 **replace_token_in_string()** 方法
- **日志级别**：DEBUG
- **作用**：追踪字符串替换过程

```python
DEBUG: "原始字符串长度: {len}"
DEBUG: "替换槽位数量: {count}"
DEBUG: "替换后字符串长度: {len}"
```

---

### 3️⃣ **agent.py** - Agent核心模块

#### 📍 **__init__()** 初始化方法
- **日志级别**：DEBUG / INFO
- **作用**：追踪Agent初始化过程

```python
DEBUG: "开始初始化Agent"
DEBUG: "初始化LLM模型"
INFO: "LLM模型加载完成"
DEBUG: "加载向量数据库"
INFO: "向量数据库加载完成: ./data/db"
DEBUG: "初始化证据追踪列表"
DEBUG: "构建工具列表"
INFO: "工具列表构建完成: {count}个工具"
DEBUG: "系统提示长度: {length}字符"
DEBUG: "创建Agent实例"
INFO: "Agent初始化完成"
```

#### 📍 **_build_tools()** 工具构建方法
- **日志级别**：DEBUG / INFO
- **作用**：追踪工具注册过程

```python
DEBUG: "开始构建工具列表"
INFO: "generic_tool - 调用通用工具: {query}"
INFO: "retrival_tool - 调用检索工具: {query}"
INFO: "graph_tool - 调用图谱工具: {query}"
INFO: "search_tool - 调用搜索工具: {query}"
INFO: "工具构建完成: ['generic_tool', 'retrival_tool', 'graph_tool', 'search_tool']"
```

#### 📍 **generic_func()** 通用对话方法
- **日志级别**：DEBUG / INFO / ERROR
- **作用**：追踪通用对话处理

```python
DEBUG: "处理通用问题: {query}"
DEBUG: "通用提示词模板加载完成"
DEBUG: "LLM链创建完成"
INFO: "通用回答生成成功"
DEBUG: "答案长度: {length}字符"
ERROR: "通用对话失败: {error}"
```

#### 📍 **retrival_func()** 向量检索方法
- **日志级别**：DEBUG / INFO / ERROR
- **作用**：追踪药品说明书检索过程

```python
INFO: "开始向量检索: {query}"
DEBUG: "执行相似度搜索，k=10"
DEBUG: "检索到 {count} 个文档"
DEBUG: "相关性阈值 0.2，过滤后剩余 {count} 个"
DEBUG: "已保存 {count} 条证据"
DEBUG: "检索提示词加载完成"
DEBUG: "输入参数准备完成"
INFO: "检索回答生成成功"
DEBUG: "答案长度: {length}字符"
ERROR: "向量检索失败: {error}"
```

#### 📍 **query()** 核心查询方法
- **日志级别**：DEBUG / INFO / WARNING / ERROR
- **作用**：追踪整个查询流程

```python
INFO: "收到用户查询: {query}"
DEBUG: "return_context参数: {bool}"

# 执行过程
DEBUG: "证据列表已重置"
DEBUG: "用户消息已添加，当前历史长度: {len}"
DEBUG: "历史长度已截断至4条"

# Agent调用
DEBUG: "开始调用Agent"
DEBUG: "Agent调用成功(模式1)"
WARNING: "Agent调用失败，尝试备用模式: {error}"
DEBUG: "Agent调用成功(模式2)"

# 答案提取
DEBUG: "开始提取答案"
DEBUG: "提取到 {count} 条消息"
DEBUG: "从字典消息中提取答案"
INFO: "答案提取完成，长度: {length} 字符"
DEBUG: "AI回答已添加，当前历史长度: {len}"

# 返回结果
INFO: "返回答案和上下文，上下文数量: {count}"
INFO: "返回最终答案"
ERROR: "查询处理失败: {error}"
```

---

### 4️⃣ **service.py** - 服务层模块

#### 📍 **__init__()** 初始化方法
- **日志级别**：DEBUG / INFO
- **作用**：追踪Service初始化

```python
DEBUG: "开始初始化Service"
DEBUG: "创建Agent实例"
INFO: "Service初始化完成"
```

#### 📍 **get_summary_message()** 摘要方法
- **日志级别**：DEBUG / INFO / ERROR
- **作用**：追踪对话历史摘要生成

```python
DEBUG: "开始生成摘要"
DEBUG: "当前消息: {message}"
DEBUG: "历史记录: {count} 条"
DEBUG: "加载LLM模型"
DEBUG: "加载摘要提示词"
DEBUG: "压缩后历史: {history[:100]}..."
INFO: "摘要生成完成"
DEBUG: "摘要长度: {length} 字符"
ERROR: "摘要生成失败: {error}"
```

#### 📍 **answer()** 核心回答方法
- **日志级别**：DEBUG / INFO / ERROR
- **作用**：追踪问题处理流程

```python
INFO: "开始处理问题: {message}"
DEBUG: "历史记录: {count} 条"
DEBUG: "需要摘要历史对话"
DEBUG: "摘要后消息: {message}"
DEBUG: "委托Agent处理"
INFO: "问题处理完成"
ERROR: "问题处理失败: {error}"
```

---

### 5️⃣ **app.py** - 应用入口模块

#### 📍 **doctor_bot()** 聊天机器人函数
- **日志级别**：DEBUG / INFO / ERROR
- **作用**：追踪用户交互流程

```python
DEBUG: "收到消息: {message}"
DEBUG: "历史记录: {count} 条"
DEBUG: "创建Service实例"
INFO: "开始处理问题"
INFO: "问题处理完成"
DEBUG: "回答长度: {length} 字符"
ERROR: "处理失败: {error}"
```

#### 📍 **主程序入口**
- **日志级别**：DEBUG / INFO / ERROR
- **作用**：追踪应用启动

```python
INFO: "启动医疗问诊机器人应用"
DEBUG: "Gradio配置: max_width=850px"
INFO: "应用启动成功"
ERROR: "应用启动失败: {error}"
```

---

### 6️⃣ **data_process.py** - 数据处理模块

#### 📍 **doc2vec()** 向量化函数
- **日志级别**：DEBUG / INFO / WARNING / ERROR
- **作用**：追踪文档向量化过程

```python
INFO: "开始文档向量化"

# 文本分割
DEBUG: "创建文本分割器"
DEBUG: "分割参数: chunk_size=500, overlap=80"

# 文件处理
INFO: "输入目录: {path}"
DEBUG: "处理文件 {count}: {file}"
DEBUG: "文件扩展名: {ext}"
DEBUG: "使用CSV加载器"
DEBUG: "使用PDF加载器"
DEBUG: "使用TXT加载器"
DEBUG: "文件 {file} 分割为 {count} 个文档片段"
INFO: "文件处理完成，共 {count} 个文件，{count} 个文档片段"

# 向量化存储
DEBUG: "开始向量化存储"
INFO: "向量化存储完成"
WARNING: "没有找到可处理的文档"
ERROR: "向量化失败: {error}"
```

#### 📍 **主程序入口**
- **日志级别**：DEBUG / INFO / ERROR
- **作用**：追踪程序执行

```python
INFO: "启动文档向量化程序"
INFO: "文档向量化完成"
ERROR: "文档向量化失败: {error}"
```

## 🚀 使用方法

### 1️⃣ **基本使用**

```bash
# 运行应用（自动生成日志）
python app.py

# 处理数据（自动生成日志）
python data_process.py
```

### 2️⃣ **查看日志**

```bash
# 查看实时日志
tail -f logs/medical_qa_*.log

# 查看最新日志
ls -lt logs/ | head -5

# 过滤特定级别日志
grep "ERROR" logs/medical_qa_*.log
grep "WARNING" logs/medical_qa_*.log
```

### 3️⃣ **程序内使用**

```python
from logger_config import logger_agent  # 导入对应模块的日志记录器

# 在代码中添加自定义日志
logger_agent.debug("这是一条DEBUG日志")
logger_agent.info("这是一条INFO日志")
logger_agent.warning("这是一条WARNING日志")
logger_agent.error("这是一条ERROR日志")
```

## 🔧 自定义配置

### 1️⃣ **修改日志级别**

编辑 `logger_config.py`：

```python
# 修改特定模块的日志级别
logger_agent = setup_logger('Agent', 'DEBUG')      # 改为DEBUG
logger_service = setup_logger('Service', 'DEBUG')  # 改为DEBUG
```

### 2️⃣ **添加新的日志记录器**

```python
# 在 logger_config.py 中添加
logger_custom = setup_logger('CustomModule', 'INFO')

# 在其他文件中使用
from logger_config import logger_custom
logger_custom.info("自定义模块日志")
```

### 3️⃣ **禁用文件日志**

```python
def setup_logger(name: str, level: str = 'INFO', file_logging: bool = False):
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    if logger.handlers:
        return logger

    formatter = logging.Formatter('%(asctime)s - [%(name)s] - %(levelname)s - %(message)s')

    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 条件性添加文件处理器
    if file_logging:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_handler = logging.FileHandler(f'logs/medical_qa_{timestamp}.log')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
```

## 📈 性能建议

### 1️⃣ **生产环境配置**

```python
# 推荐的生产环境配置
LOG_LEVELS = {
    'Agent': 'INFO',      # 保留关键流程
    'Service': 'INFO',
    'Utils': 'WARNING',   # 减少工具函数日志
    'App': 'INFO',
    'DataProcess': 'INFO'
}
```

### 2️⃣ **开发环境配置**

```python
# 推荐的开发环境配置
LOG_LEVELS = {
    'Agent': 'DEBUG',     # 详细调试信息
    'Service': 'DEBUG',
    'Utils': 'DEBUG',
    'App': 'DEBUG',
    'DataProcess': 'DEBUG'
}
```

## 🎯 故障排查指南

### 1️⃣ **常见问题**

| 问题现象 | 排查方法 | 相关日志 |
|----------|----------|----------|
| 模型加载失败 | 查看 ERROR 日志 | `logger_utils.error` |
| 数据库连接失败 | 查看 ERROR 日志 | `get_neo4j_conn()` |
| 向量检索无结果 | 查看 DEBUG 日志 | `retrival_func()` |
| Agent调用失败 | 查看 WARNING 日志 | `query()` |
| 应用启动失败 | 查看 ERROR 日志 | `app.py` 主程序 |

### 2️⃣ **日志分析示例**

```bash
# 分析错误
grep "ERROR" logs/medical_qa_*.log | head -10

# 分析性能
grep "完成" logs/medical_qa_*.log

# 分析工具调用
grep "调用.*工具" logs/medical_qa_*.log

# 分析检索过程
grep "retrival_func" logs/medical_qa_*.log
```

## 📝 注意事项

1. **日志文件会持续增长**，建议定期清理或配置日志轮转
2. **生产环境避免使用DEBUG级别**，防止性能影响
3. **敏感信息不要写入日志**，如API密钥、密码等
4. **日志格式统一**，便于自动化分析

---

## 🎉 总结

通过这套完整的日志系统，你可以：

- ✅ **实时监控**系统运行状态
- ✅ **快速定位**问题原因
- ✅ **追踪性能**瓶颈
- ✅ **分析用户行为**
- ✅ **优化系统性能**

建议在开发和测试阶段开启DEBUG级别，生产环境使用INFO级别，这样既能保证问题可追踪，又不会影响系统性能。
