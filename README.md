# RAG 项目

这是一个学习型 RAG 项目。目前已包含文档加载、文本切分、Embedding、本地 JSON 索引、向量检索、Prompt 构建和基于 Anthropic 的问答生成。

## 项目结构

```text
.
├── ingest.py             # 构建本地 index 的命令入口
├── retrieve.py           # 从本地 index 检索 Top K chunks
├── evaluate_retrieval.py # 计算 Retriever 的 Recall@1 和 Recall@3
├── evaluation_questions.json # 10 条 retrieval 测试问题
├── src/
│   └── rag_app/          # 应用源代码包
│       ├── __init__.py
│       ├── __main__.py   # `python -m src.rag_app` 的模块入口
│       ├── app.py        # 最小启动函数
│       ├── chunker.py    # 按字符切分文档
│       ├── config.py     # 集中管理模型配置
│       ├── document_loader.py # 加载 txt 和 md 文件
│       ├── embedding.py  # 文本向量化
│       ├── indexer.py    # 编排并保存本地 JSON index
│       ├── retriever.py  # 计算相似度并返回 Top K chunks
│       ├── prompt_builder.py # 用问题和检索结果构建 Prompt
│       ├── generator.py  # 调用 Anthropic 生成回答
│       └── rag_service.py # 编排完整 RAG 问答流程
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

命令依次执行 loader → chunker → embedding，并生成 `data/processed/index.json`。该文件是普通 JSON 数组，每条记录包含：

```json
{
  "source": "rag_notes.md",
  "chunk_id": "rag_notes.md#chunk-0",
  "text": "...",
  "embedding": [0.01, 0.02]
}
```

运行完成后，终端会显示文件数量、chunk 数量和 embedding 数量。默认 chunk 大小为 500 个字符，相邻 chunk 重叠 50 个字符。

## 检索 Top K chunks

先构建 index，然后在项目根目录执行：

```powershell
python retrieve.py "什么是 RAG？" --top-k 3
```

`--top-k` 可配置返回数量，默认值为 3；`--index` 可指定其他 index 文件。命令只执行 question embedding、余弦相似度计算和排序，不调用 LLM。输出是按 `score` 降序排列的 JSON：

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

## 运行测试

使用 `uv`：

```powershell
uv run --no-project --python 3.12 --with-requirements requirements.txt python -m unittest discover -s tests -v
```

或使用已安装并激活的 Python：

```powershell
python -m unittest discover -s tests -v
```
