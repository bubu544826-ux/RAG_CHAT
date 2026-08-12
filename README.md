# RAG 项目

这是一个学习型 RAG 项目。目前已包含文档加载、文本切分、Embedding、本地 Chroma vector store、向量检索、Prompt 构建和基于 Anthropic 的问答生成。

## 项目结构

```text
.
├── ingest.py             # 构建本地 index 的命令入口
├── retrieve.py           # 从本地 index 检索 Top K chunks
├── evaluate_retrieval.py # 计算 Retriever 的 Recall@1 和 Recall@3
├── evaluate_rag.py       # 端到端评估 retrieval、回答事实和拒答
├── evaluation_questions.json # 10 条 retrieval 测试问题
├── evaluation_cases.json # 端到端 RAG 测试数据
├── src/
│   └── rag_app/          # 应用源代码包
│       ├── __init__.py
│       ├── __main__.py   # `python -m src.rag_app` 的模块入口
│       ├── app.py        # 最小启动函数
│       ├── api.py        # FastAPI /chat 接口
│       ├── chunker.py    # 按字符切分文档
│       ├── config.py     # 集中管理模型配置
│       ├── document_loader.py # 加载 txt 和 md 文件
│       ├── embedding.py  # 文本向量化
│       ├── indexer.py    # 编排并保存本地 Chroma index
│       ├── retriever.py  # 查询 Chroma 并返回 Top K chunks
│       ├── prompt_builder.py # 用问题和检索结果构建 Prompt
│       ├── generator.py  # 调用 Anthropic 生成回答
│       ├── rag_service.py # 编排完整 RAG 问答流程
│       └── web/          # 简单 Web UI（HTML、CSS、JavaScript）
├── data/
│   ├── raw/              # 原始数据（默认不提交到 Git）
│   └── processed/        # 处理后的数据（默认不提交到 Git）
├── tests/                # 自动化测试
├── .env.example          # 环境变量示例，不包含真实密钥
├── .gitignore            # Git 忽略规则
├── requirements.txt      # Python 第三方依赖
└── README.md             # 项目说明
```

## 环境要求

- Python 3.10 或更高版本

## 安装与运行

### 使用 uv（当前环境推荐）

当前机器已经安装 `uv` 并有可用的 Python 3.12。在项目根目录可以直接执行：

```powershell
uv run --no-project --python 3.12 python -m src.rag_app
```

### 使用已安装的 Python

如果 `python --version` 能正常输出 Python 3.10 或更高版本，则执行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
python -m src.rag_app
```

预期输出：

```text
RAG project initialized successfully.
```

## 查看 Embedding 类型和维度

测试脚本会在首次运行时从 Hugging Face 下载配置的模型。输入一句话后，脚本只显示 vector 类型和维度，不会打印完整向量：

```powershell
python -m scripts.test_embedding
```

如果使用 `uv` 且尚未安装依赖，可以直接执行：

```powershell
uv run --no-project --python 3.12 --with-requirements requirements.txt python -m scripts.test_embedding
```

默认模型在 `src/rag_app/config.py` 中集中配置，也可以在启动进程前通过 `EMBEDDING_MODEL_NAME` 环境变量覆盖。

## 构建本地 index

把 `.txt` 或 `.md` 文件放入 `data/raw/`，然后在项目根目录执行：

```powershell
python ingest.py
```

命令依次执行 loader → chunker → embedding，并在 `data/processed/index.json`
目录中生成持久化 Chroma vector store。目录名保留 `index.json` 是为了兼容现有
Retriever 和 RAGService 调用；它不再是 JSON 文件。

Chroma collection 名为 `rag_chunks`，使用 cosine distance。每个 chunk 保存为：

```json
{
  "id": "rag_notes.md#chunk-0",
  "document": "...",
  "metadata": {
    "source": "rag_notes.md",
    "chunk_id": "rag_notes.md#chunk-0"
  },
  "embedding": [0.01, 0.02]
}
```

首次迁移时，如果路径上存在旧 `index.json` 文件，ingestion 会把它改名为
`index.json.legacy` 后再建立 Chroma 目录。重复运行 ingestion 会完整重建
`rag_chunks` collection，避免已经删除的文档残留在索引中。

运行完成后，终端会显示文件数量、chunk 数量和 embedding 数量。默认 chunk 大小为 500 个字符，相邻 chunk 重叠 50 个字符。

## 检索 Top K chunks

先构建 index，然后在项目根目录执行：

```powershell
python retrieve.py "什么是 RAG？" --top-k 3
```

`--top-k` 可配置返回数量，默认值为 3；`--index` 可指定其他 Chroma index
目录。命令只执行 question embedding 和 Chroma cosine 查询，不调用 LLM。
Retriever 会把 Chroma distance 转换成原接口使用的 cosine similarity score，
输出仍是按 `score` 降序排列的 JSON：

```json
[
  {
    "text": "...",
    "source": "rag_notes.md",
    "chunk_id": "rag_notes.md#chunk-0",
    "score": 0.82
  }
]
```

## 运行 Retrieval Evaluation

先构建 index，然后运行：

```powershell
python evaluate_retrieval.py
```

脚本读取 `evaluation_questions.json` 中的 10 个 `question` / `expected_source`
测试对。每个问题只检索一次 Top 3，检查正确 source 是否出现在第 1 条和前 3 条，
最后输出 `Recall@1` 和 `Recall@3`。可以用 `--questions` 和 `--index` 指定其他文件。

## 使用 RAGService 问答

先构建本地 index，然后把 Anthropic API key 写入项目根目录的 `.env`：

```dotenv
ANTHROPIC_API_KEY=your-anthropic-api-key
```

使用 `uv run` 时通过 `--env-file .env` 加载该文件：

```powershell
uv run --env-file .env --no-project --python 3.12 --with-requirements requirements.txt python -c "from src.rag_app.rag_service import RAGService; print(RAGService().ask('What is RAG?'))"
```

然后可以在 Python 中调用：

```python
from src.rag_app.rag_service import RAGService

service = RAGService()
answer = service.ask("什么是 RAG？")
print(answer)
```

`ask(question)` 会依次执行 `retrieve -> build_prompt -> generate`。默认模型由
`src/rag_app/config.py` 集中管理，也可以通过 `ANTHROPIC_MODEL_NAME` 环境变量覆盖。

## 运行端到端 RAG Evaluation

先构建 index 并配置 `ANTHROPIC_API_KEY`，然后运行：

```powershell
python evaluate_rag.py --top-k 3
```

脚本读取 `evaluation_cases.json`。有答案样本使用
`{question, expected_source, expected_answer_keywords}`；无答案样本把
`expected_source` 设为 `null`、关键字设为空数组。每条问题只调用一次完整 RAG
流程，并计算：

- `retrieval_recall_at_3`：正确来源是否出现在 Top 3，仅统计有答案问题。
- `answer_keyword_pass_rate`：回答是否包含该样本的全部关键事实。
- `no_answer_refusal_rate`：无答案问题是否明确使用“无法回答”等拒答表达。

汇总结果会打印到终端，逐题明细默认写入 `evaluation_report.json`。可以使用
`--cases`、`--index`、`--top-k` 和 `--output` 修改输入、K 值和报告路径。

## 启动 FastAPI

先构建本地 index，并在 `.env` 中配置 `ANTHROPIC_API_KEY`。使用 `uv` 启动：

```powershell
uv run --env-file .env --no-project --python 3.12 --with-requirements requirements.txt uvicorn src.rag_app.api:app --host 127.0.0.1 --port 8000
```

如果已经安装依赖并激活虚拟环境，则运行：

```powershell
uvicorn src.rag_app.api:app --host 127.0.0.1 --port 8000
```

启动后打开 `http://127.0.0.1:8000/` 即可使用 Web UI；交互式 API 文档仍可在
`http://127.0.0.1:8000/docs` 查看。

接口接收问题，并原样返回 `RAGService.ask()` 生成的答案和来源信息：

```http
POST /chat
Content-Type: application/json

{"question": "什么是 RAG？"}
```

## 运行测试

使用 `uv`：

```powershell
uv run --no-project --python 3.12 --with-requirements requirements.txt python -m unittest discover -s tests -v
```

或使用已安装并激活的 Python：

```powershell
python -m unittest discover -s tests -v
```
