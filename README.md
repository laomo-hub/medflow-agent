# MedFlow Agent（医流智能助手）

面向医疗场景的智能体开发项目。本分支先整理基础设施、运行配置和初始化入口，后续 PR 逐步加入专项 Agent 与 Chat API。

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

访问 <http://localhost:8000/docs>，通过 `GET /health`、`GET /health/deps` 检查服务及数据库连接。Chat API 将在后续编排 PR 接入。

## 聊天模型兼容配置

聊天模型为 `deepseek-flash`，显式关闭 thinking。Supervisor 与摘要模型保持温度未指定，保留原工具绑定行为；只在具体专项模型需要时单独指定温度。

## 智慧问诊内核

问诊内核位于 `src/agents/inquiry/`，包含患者上下文、症状标准化、图谱候选、多轮状态与挂号模拟移交。

初始化症状向量索引：`python scripts/init_symptom_index.py`，每批保持 **20** 条；该命令调用配置的向量化服务。此 PR 提供工作流模块，统一 Chat 路由在后续 PR 接入。

## 文档入库与检索

`src/agents/knowledge/` 提供 MinerU 解析、文档分块、Milvus 入库、文档/图谱检索及检索增强模块。使用解析服务前配置 `MINERU_API_URL`、`MINERU_BACKEND` 和 `MINERU_TIMEOUT`。检索模块可单独导入，专项 Agent 将在下一 PR 接入。

## 来源与许可证

项目基于[原始课程项目](https://gitee.com/leifengyang/tiangong-agent)学习与扩展；许可证见 [LICENSE](LICENSE)。
