# SmartQ

> 基于 RAG 与多 Agent 架构的企业级智能问答系统

SmartQ 面向企业知识库问答与智能运维分析场景，整合内部文档检索、Prometheus 告警查询、日志分析、工具调用与多 Agent 协作能力。系统支持通过自然语言完成知识问答、告警分析、日志排查和处置建议生成，适合作为 Spring AI Alibaba、RAG、Agent 工具调用与 AIOps 场景的工程实践项目。

**作者**: mengyilin

## 功能特性

- **RAG 知识库问答**: 支持文档上传、Markdown 分片、DashScope Embedding 向量化、Milvus 向量存储与检索增强生成。
- **混合检索**: 结合向量检索、BM25 关键词检索、Query Rewrite、Multi-Query 和 RRF 融合排序，提升复杂问题与关键词场景下的召回效果。
- **可信生成**: 将检索结果组织为带编号引用的上下文，引导模型逐步分析、标注来源，并在缺少依据时拒答。
- **ReAct Agent 对话**: 基于 Spring AI Alibaba Agent Framework 构建智能对话 Agent，支持自动工具调用。
- **Multi-Agent AIOps**: 采用 Supervisor / Planner / Executor 协作模式，自动执行告警读取、任务拆解、日志/指标查询、知识库检索和报告生成。
- **工具体系**: 封装文档检索、Prometheus 告警查询、日志查询、时间查询等工具，并支持 MCP 扩展外部工具。
- **流式体验**: 基于 SSE 实现 AI 回答和 AIOps 分析结果的实时输出。
- **会话管理**: 基于 sessionId 维护短期多轮对话上下文。

## 技术栈

| 类型 | 技术 |
| --- | --- |
| 后端框架 | Java 17, Spring Boot 3.2 |
| AI 框架 | Spring AI, Spring AI Alibaba Agent Framework |
| 模型服务 | Alibaba DashScope, Qwen, text-embedding-v4 |
| 向量数据库 | Milvus, Attu |
| 检索增强 | RAG, BM25, RRF, Query Rewrite, Multi-Query |
| Agent | ReAct Agent, Supervisor Agent, Planner / Executor |
| 通信与工具 | SSE, MCP, Spring AI Tool |
| 构建工具 | Maven, Docker Compose |

## 项目结构

```text
SmartQ/
├── aiops-docs/                         # 示例运维知识库文档
├── src/main/java/org/example/
│   ├── agent/tool/                     # Agent 工具
│   │   ├── DateTimeTools.java
│   │   ├── InternalDocsTools.java
│   │   ├── QueryLogsTools.java
│   │   └── QueryMetricsTools.java
│   ├── client/
│   │   └── MilvusClientFactory.java    # Milvus collection 初始化
│   ├── config/                         # 配置类
│   ├── controller/
│   │   ├── ChatController.java         # 聊天、RAG、AIOps 接口
│   │   ├── FileUploadController.java   # 文件上传与索引触发
│   │   └── MilvusCheckController.java  # Milvus 健康检查
│   ├── dto/
│   └── service/
│       ├── AiOpsService.java           # 多 Agent AIOps 编排
│       ├── ChatService.java            # ReAct Agent 对话
│       ├── RagService.java             # RAG 生成链路
│       ├── VectorIndexService.java     # 文档入库与向量索引
│       ├── VectorSearchService.java    # Milvus 向量检索
│       ├── HybridSearchService.java    # 混合召回与 RRF 融合
│       ├── BM25SearchService.java      # 内存 BM25 关键词检索
│       └── QueryRewriteService.java    # 查询改写与 Multi-Query
├── src/main/resources/
│   ├── static/                         # SmartQ 前端页面
│   └── application.yml                 # 应用配置
├── vector-database.yml                 # Milvus/Etcd/MinIO/Attu
├── Makefile
└── pom.xml
```

## 核心流程

### 1. 知识入库

```text
上传 .md/.txt 文档
 -> Markdown 标题与段落分片
 -> DashScope text-embedding-v4 向量化
 -> 写入 Milvus collection: biz
 -> 同步写入内存 BM25 索引
```

### 2. RAG 问答

```text
用户问题
 -> 会话上下文读取
 -> Query Rewrite + Multi-Query
 -> 向量检索 + BM25 检索
 -> RRF 融合排序
 -> 组装带引用编号的上下文
 -> DashScope LLM 流式生成
```

### 3. AIOps 多 Agent 分析

```text
AI Ops 任务
 -> Supervisor 路由
 -> Planner 拆解/再规划
 -> Executor 调用工具
 -> 汇总指标、日志、知识库证据
 -> 输出告警分析报告
```

## 快速开始

### 环境要求

- JDK 17+
- Maven 3.8+
- Docker / Docker Compose
- DashScope API Key

### 1. 配置环境变量

Windows PowerShell:

```powershell
$env:DASHSCOPE_API_KEY="your-dashscope-api-key"
```

macOS/Linux:

```bash
export DASHSCOPE_API_KEY="your-dashscope-api-key"
```

### 2. 启动 Milvus

```bash
docker compose -f vector-database.yml up -d
```

启动后可访问：

- Milvus: `localhost:19530`
- Attu: `http://localhost:8000`
- MinIO Console: `http://localhost:9001`

### 3. 启动后端服务

```bash
mvn spring-boot:run
```

服务默认端口：

```text
http://localhost:9900
```

### 4. 访问前端

```text
http://localhost:9900
```

### 5. 上传知识库文档

```bash
curl -X POST http://localhost:9900/api/upload \
  -F "file=@aiops-docs/cpu_high_usage.md"
```

也可以在页面中点击上传文件。

## API 示例

### 普通对话

```bash
curl -X POST http://localhost:9900/api/chat \
  -H "Content-Type: application/json" \
  -d "{\"Id\":\"demo-session\",\"Question\":\"CPU 使用率过高怎么排查？\"}"
```

### 流式对话

```bash
curl -X POST http://localhost:9900/api/chat_stream \
  -H "Content-Type: application/json" \
  -d "{\"Id\":\"demo-session\",\"Question\":\"HighMemoryUsage 告警怎么处理？\"}"
```

### 专用 RAG 流式问答

```bash
curl -X POST http://localhost:9900/api/rag_stream \
  -H "Content-Type: application/json" \
  -d "{\"Id\":\"rag-session\",\"Question\":\"服务 OOM 应该怎么排查？\"}"
```

### AIOps 自动分析

```bash
curl -X POST http://localhost:9900/api/ai_ops
```

### 清空会话

```bash
curl -X POST http://localhost:9900/api/chat/clear \
  -H "Content-Type: application/json" \
  -d "{\"Id\":\"demo-session\"}"
```

### Milvus 健康检查

```bash
curl http://localhost:9900/milvus/health
```

## 重要配置

`src/main/resources/application.yml` 中的主要配置：

```yaml
server:
  port: 9900

milvus:
  host: localhost
  port: 19530

dashscope:
  api:
    key: ${DASHSCOPE_API_KEY:}
  embedding:
    model: text-embedding-v4

document:
  chunk:
    max-size: 800
    overlap: 100

rag:
  top-k: 3
  model: "qwen3-max"
  query-rewrite-enabled: true
  multi-query-count: 3
  hybrid-enabled: true
  bm25:
    enabled: true
  similarity-threshold: 0.5

prometheus:
  mock-enabled: true

cls:
  mock-enabled: true
```

默认情况下，Prometheus 与 CLS 日志查询使用 Mock 数据，便于本地演示 AIOps 流程。如果要接入真实环境：

1. 将 `prometheus.mock-enabled` 改为 `false`，并配置可访问的 Prometheus 地址。
2. 将 `cls.mock-enabled` 改为 `false`。
3. 启用 MCP 客户端并配置可用的 MCP 服务。
4. 确保云日志账号具备对应的日志查询权限。

## 当前限制

- BM25 索引当前存储在内存中，服务重启后需要重新上传或重建文档索引。
- AIOps 默认使用 Mock 告警和日志数据，真实环境需要配置 Prometheus、MCP 与云日志权限。
- 会话记忆为内存短期窗口，服务重启后不会持久化。



