# MedFlow Agent（医流智能助手）

面向医疗场景的智能体开发项目。整合智慧问诊、知识问答与 Supervisor 分流；报告、药物及运营 Worker 目前保留模拟能力。

仓库：<https://github.com/laomo-hub/medflow-agent>

## 安装与配置

```powershell
git clone https://github.com/laomo-hub/medflow-agent.git
cd medflow-agent
conda create -n medflow python=3.13
conda activate medflow
pip install -r requirements.txt
Copy-Item .env.example .env
```

编辑 `.env`，填入自己的 API Key 和服务地址。`.env`、`docker/`、`data/`、`.backups/`、本地测试及迁移目录不提交 Git。保留的历史提交可能仍包含此前版本的数据文件。

## 基础设施

```powershell
docker compose up -d
docker compose ps
```

Compose 项目为 `medflow-agent`，容器使用 `medflow-` 前缀。PostgreSQL、Redis Stack、Neo4j、Milvus 及 MinIO 的持久化数据位于项目 `docker/` 目录；Attu 通过 Compose 网络内的 `milvus:19530` 连接 Milvus。

| 服务 | 地址 |
|---|---|
| PostgreSQL | localhost:5432 |
| Redis / RedisInsight | localhost:6379 / http://localhost:8001 |
| MinIO | http://localhost:9000 / http://localhost:9001 |
| Milvus / Attu | localhost:19530 / http://localhost:3000 |
| Neo4j | http://localhost:7474 / bolt://localhost:7687 |

停止服务使用 `docker compose down`。迁移项目目录时保留 `docker/`，并从新目录重建容器挂载。

## 首次建表与基础数据

仓库不包含本地 `alembic/` 迁移脚本；首次部署使用模型生成表结构：

```powershell
python -m scripts.init_schema
python -m scripts.init_schema --apply
```

第一条仅打印 PostgreSQL 建表 SQL，第二条创建尚不存在的表。此入口不修改已有表的字段，也不导入业务数据；已有数据库升级应另行准备结构变更方案。

自行准备课程来源数据并放到 `data/raw/medical.json`，再按需执行：

```powershell
python scripts/init_postgres.py
python scripts/init_neo4j.py
```

`data/` 是本地输入目录，不随仓库分发。

## 启动与检查

```powershell
uvicorn src.main:app --host 127.0.0.1 --port 8000 --reload
```

访问 <http://localhost:8000/docs>，通过 `GET /health`、`GET /health/deps` 检查服务及数据库连接。Chat API 已接入：`POST /api/v1/chat` 与 `POST /api/v1/chat/stream`。

## 聊天模型兼容配置

聊天模型为 `deepseek-flash`，显式关闭 thinking。Supervisor 与摘要模型保持温度未指定，保留原工具绑定行为；只在具体专项模型需要时单独指定温度。

## 智慧问诊内核

问诊内核位于 `src/agents/inquiry/`，包含患者上下文、症状标准化、图谱候选、多轮状态与挂号模拟移交。

初始化症状向量索引：`python scripts/init_symptom_index.py`，每批保持 **20** 条；该命令调用配置的向量化服务。问诊内核由统一 Chat 路由接入，活跃问诊根据 Redis 状态继续执行。

## 文档入库与检索

`src/agents/knowledge/` 提供 MinerU 解析、文档分块、Milvus 入库、文档/图谱检索及检索增强模块。使用解析服务前配置 `MINERU_API_URL`、`MINERU_BACKEND` 和 `MINERU_TIMEOUT`。检索模块可单独导入，也由知识问答 Agent 的工具调用。

## 知识问答 Agent

知识 Agent 提供文档、图谱、SQL、多通道融合与处方审核工具，并包含反馈、通知和会话辅助模块。SQL 查询需要传入数据库会话；默认未传入时会返回连接不可用提示。

新部署运行 `python -m scripts.init_schema` 可预览包括 `knowledge_feedback`、`knowledge_notifications` 的表结构，显式加 `--apply` 创建缺失表。此操作不会自动执行。

## Chat 使用与本地验证

```json
{"user_id":"1","session_id":"medflow-demo-001","message":"我肚子有点不舒服"}
```

活跃问诊由路由恢复 Redis 状态继续处理，其他请求进入 Supervisor。报告、药物及运营 Worker 仍是模拟入口；不代表完整业务实现。

本地 `test/` 按项目忽略策略不随仓库发布；维护者可运行 `python -m pytest test/test_flash_thinking.py -q` 做离线回归，实际 PR 验证结果记录在 PR 说明。长期记忆样例输入保留“糖尿病史”。

## RAG 评测（独立环境）

`src/rag/evaluation/` 提供答案相关性、上下文相关性和有据性指标，以及 `RAGEngine` 的检索、生成和查询追踪包装。评测模块由本地评测调用方使用；现有知识 Agent 和上传流程保持原有行为。

评测依赖单独锁定在 `requirements-eval.txt`，验证目标为 Windows x64 / Python 3.13。该清单包含运行及传递依赖；请在独立环境安装，不与应用的 `requirements.txt` 混装：

```powershell
conda create -n medflow-eval python=3.13
conda activate medflow-eval
python -m pip install -r requirements-eval.txt
python -m pip check
```

应用继续使用原 `medflow` 环境。独立评测环境使用兼容 TruLens 的 Plotly、Rich、psutil 和 protobuf 版本，应用依赖清单保持不变。

在项目根目录运行评测，沿用 `.env` 中的 `CHAT_MODEL=deepseek-flash` 和 `DEEPSEEK_API_KEY`。评审模型通过 DeepSeek 的 OpenAI 兼容接口调用，显式设置 `thinking.type=disabled`；评测 Provider 支持嵌套请求参数并额外放行 `extra_body`，确保该设置经过 TruLens 参数过滤后仍会发送。检索所需的向量化、重排和 Milvus 配置继续使用现有设置。

主要入口：

- `build_rag_triad_metrics(provider)`：构造三个评测指标。
- `TrackedRAG(engine)`：包装调用方创建的 `RAGEngine`，提供 `retrieve`、`generate`、`query`。
- `get_trulens_session()`：连接独立的 PostgreSQL `trulens_eval` 数据库。
- `launch_dashboard()`：启动 TruLens Dashboard，默认端口为 8501。

真实评测前需自行准备 `trulens_eval` 数据库，并确认 `.env` 中的 `DB_HOST`、`DB_PORT`、`DB_USER`、`DB_PASSWORD` 可访问它。首次连接可能由 TruLens 初始化其记录表；上述安装命令不创建评测数据库。完成准备后可显式启动 Dashboard：

```powershell
python -c "from src.rag.evaluation.trulens_config import launch_dashboard; launch_dashboard()"
```

题集及本地准备脚本继续留在本地，`data/` 保持忽略。本模块不自动运行题集、调用模型、导入数据或启动 Dashboard；实际评测的模型调用与记录写入由本地调用方显式触发。返回应用开发时使用 `conda activate medflow`。

## 来源与许可证

项目基于[原始课程项目](https://gitee.com/leifengyang/tiangong-agent)学习与扩展；许可证见 [LICENSE](LICENSE)。
