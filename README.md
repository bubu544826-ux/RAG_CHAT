# RAG 项目

这是一个学习型 RAG 项目。目前已包含文档加载、文本切分、Embedding、本地 Chroma vector store、向量检索、Prompt 构建和基于 Anthropic 的问答生成。

## 项目结构

```text
.
├── ingest.py             # 构建本地 index 的命令入口
├── retrieve.py           # 从本地 index 检索 Top K chunks
├── evaluate_retrieval.py # 计算 Retriever 的 Recall、Precision、MRR 和 NDCG
├── evaluate_rag.py       # 端到端评估 retrieval、回答事实和拒答
├── RAG_test.json         # 20 条带相关 chunk 标注的 retrieval 测试样本
├── evaluation_questions.json # 旧版 source 级 retrieval 测试问题
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

当前机器已经安装 `uv` 并有可用的 Python 3.12。建议先建立一个持久化的 `.venv`，
避免每次运行都重新解析 `--with-requirements` 依赖：

```powershell
uv venv --python 3.12
uv pip install -r requirements.txt
uv run python -m src.rag_app
```

也可以不建立 `.venv`，每次使用临时环境：

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

## 把 Embedding 画成 3D 向量

embedding 有一千多维，无法直接观察。可视化脚本先用 PCA 找出方差最大的 3 个方向，把向量投影到这 3 维再画成从原点出发的箭头：

```powershell
python -m scripts.visualize_embeddings
```

如果使用 `uv` 且尚未安装依赖，可以直接执行：

```powershell
uv run --no-project --python 3.12 --with-requirements requirements.txt python -m scripts.visualize_embeddings
```

默认使用内置的示例句子（动物、编程、天气三组），图片保存到 `data/processed/embeddings_3d.png`。常用参数：

| 参数 | 说明 |
| --- | --- |
| `--input sentences.txt` | 改用自己的句子，每行一句，至少 3 句 |
| `--output my_plot.png` | 修改图片保存路径 |
| `--show` | 保存后打开交互窗口，可以旋转查看 |

坐标轴标题中的百分比是该主成分保留的信息比例。三个百分比之和通常远小于 100%，说明这张图只是高维空间的一个投影，图上的距离不等于真实的向量距离。

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

Indexer 会先收集全部 chunk，再通过 `embed_texts` 一次性分批向量化（默认
`batch_size=32`），而不是每个 chunk 调用一次模型。`ingest.py` 会显示批次进度条。

## 让 Embedding 使用 GPU

Embedding 是 ingestion 中最慢的一步。PyPI 在 Windows 上默认安装的 `torch` 是
**CPU-only** 版本，即使机器有 NVIDIA 显卡也不会使用。检查当前状态：

```powershell
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

如果输出中版本号带 `+cpu` 或 `is_available()` 为 `False`，安装带 CUDA 的版本。
本机的 Quadro P4000 属于 Pascal 架构（compute capability 6.1），CUDA 12.8 以后
已不再支持该架构，因此需要指定 cu126：

```powershell
uv pip install torch --index-url https://download.pytorch.org/whl/cu126 --reinstall-package torch
```

### 还需要足够新的显卡驱动

只装 CUDA 版 torch 还不够，**显卡驱动也必须支持对应的 CUDA 版本**。用
`nvidia-smi` 查看右上角的 `CUDA Version`：

```powershell
nvidia-smi
```

本机原本是驱动 516.40（CUDA 11.7，2022 年版本），torch cu126 会报
`CUDA initialization: The NVIDIA driver on your system is too old`，
并且 `is_available()` 仍为 `False`。CUDA 12.x 需要 525 以上的驱动。

从 <https://www.nvidia.com/Download/index.aspx> 选择
`NVIDIA RTX / Quadro` → `Quadro Series` → `Quadro P4000` → `Windows 10 64-bit`
下载并安装最新驱动，然后重启。Pascal 架构的支持一直保留到 R580 驱动分支。

驱动更新后不需要重装 torch，重新执行检查命令，`is_available()` 应为 `True`。
sentence-transformers 会自动使用检测到的 GPU，不需要修改代码。

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

脚本默认读取 `RAG_test.json`，每个问题只检索一次，并按显式标注的相关
chunk 计算排名指标。默认 `K=5`，**并且默认走和 `RAGService` 完全一致的生产检索
管线**（`vector + BM25 -> RRF -> CrossEncoder rerank -> 邻居扩展`），配置来自
`config.RETRIEVAL_SETTINGS`，因此评估结果和线上问答用的是同一条管线。

```powershell
python evaluate_retrieval.py --test-set RAG_test.json --index data/processed/index.json --top-k 5

# 只评估旧的 vector-only 基线，便于和历史数字对比
python evaluate_retrieval.py --baseline
```

在当前 20 条测试样本上：

| 管线 | Recall@5 | MRR@5 | NDCG@5 |
| --- | --- | --- | --- |
| `--baseline`（vector-only） | 69.17% | 61.83% | 59.20% |
| 默认生产管线 | **84.17%** | **81.67%** | **77.53%** |

测试集是非空 JSON 数组。每条样本必须有唯一的非空 `id`、非空 `question`，
以及由唯一、非空字符串组成的 `relevant_chunk_ids`；`ground_truth`、`evidence`
等字段会保留为元数据，但 retrieval 评估不会调用 LLM，也不使用相似度阈值：

```json
{
  "id": "rag_quic_001",
  "question": "In QUIC, what is the final size of a stream?",
  "relevant_chunk_ids": ["RCF.txt#chunk-116"],
  "ground_truth": "..."
}
```

四项指标先逐题计算，再对全部问题做宏平均，并以 `0` 到 `100` 的百分数输出：

- `Recall@K = 前 K 条中的相关 chunk 数 / 该题全部标注相关 chunk 数`
- `Precision@K = 前 K 条中的相关 chunk 数 / K`。分母是 `K` 而不是标注数量，
  所以只标了 1 个相关 chunk 的题目在 `K=5` 时最高只能拿 20%。当前测试集平均
  每题 1.5 个标注，`Precision@5` 的理论上限是 30%，脚本会把这个上限一起打印，
  避免把接近上限的数字误读成失败。`NDCG@K` 做了归一化，更适合当主指标
- `MRR@K = 1 / 第一个相关 chunk 的排名`；前 K 条无命中时为 `0`
- `NDCG@K = DCG@K / IDCG@K`，其中二元相关性 `rel` 为 `0` 或 `1`，
  `DCG@K = Σ rel(rank) / log2(rank + 1)`

当前 `RAG_test.json` 的 chunk 标签依赖 `data/raw/RCF.txt` 以及
`chunk_size=500`、`overlap=50` 的切分配置。源文档或切分参数变化后，必须重新核对
`relevant_chunk_ids`，否则指标不再代表真实检索质量。`--questions` 仍作为
`--test-set` 的兼容别名保留。

### 对比完整检索管线

应用默认使用 `vector + BM25 -> RRF -> CrossEncoder rerank -> 邻居扩展`。
直接调用 `retriever.retrieve()` 时仍保持原来的 vector-only 默认行为，便于旧代码兼容；
所有生产入口（`RAGService`、`retrieve.py`、`evaluate_retrieval.py`）都通过
`retriever.production_retrieval_options()` 读取同一份配置，不会再各自漂移。

运行使用同一标签和 cutoff 的对照实验：

```powershell
python evaluate_retrieval.py --compare --output retrieval_evaluation_report.json

# 额外评测多个 reranker 模型（首次运行会下载模型）
python evaluate_retrieval.py --compare --compare-rerankers
```

#### 邻居扩展（neighbour expansion）

chunker 按固定 500 字符硬切、只重叠 50 字符，一个事实经常被切到相邻两个 chunk 里
（20 条测试样本中有 8 条的标注就是相邻 chunk 对）。因此重排之后会把命中 chunk 的
前后邻居一起纳入结果，再截断到 `top_k`。这是当前单项收益最大的改动：
`Recall@5 74.17% -> 84.17%`、`NDCG@5 71.45% -> 77.53%`。

`NEIGHBOUR_RADIUS=2` 在这个测试集上更高（`Recall@5 86.67%`、`NDCG@5 78.85%`），
但 `top_k=5` 时结果会退化成"一个命中点 ± 2 个邻居"的连续窗口，牺牲了跨小节取证的
能力，所以默认保持 `1`。

#### 关于 reranker 模型选型

在同一测试集上对比过三个 CrossEncoder（`--compare-rerankers`，CPU）：

| 模型 | NDCG@5 | median 延迟 |
| --- | --- | --- |
| `cross-encoder/ms-marco-MiniLM-L6-v2`（默认） | **77.53%** | **890ms** |
| `cross-encoder/ms-marco-MiniLM-L12-v2` | 74.68% | 1378ms |
| `BAAI/bge-reranker-base` | 78.33% | 2733ms |

L12 反而比 L6 差，bge-base 只多 0.8 个点却慢 3 倍，因此保持 L6 不变。换句话说，
排序阶段的剩余损失不是靠换更大的 reranker 能解决的：候选池里已经有 98.33% 的标注
chunk，真正的瓶颈是 500 字符硬切造成的 chunk 片段（很多 chunk 从半个单词开始），
cross-encoder 也很难对这种片段打分。下一步真正值得做的是改切分，但那会让
`RAG_test.json` 里所有 `relevant_chunk_ids` 失效，需要先重新标注。

#### 关于 query rewrite

`QUERY_REWRITE_ENABLED` 默认已改为 `false`。`rule_based_rewrite` 是关键词裁剪而不是
LLM 改写，而且 `is_precise_query()` 会在包含 `MAX_STREAM_DATA` 这类 token 的 QUIC
问题上直接短路，实测四项指标与关闭时完全一致，却让每次查询多花约 500ms。
代码路径和环境变量都保留，换语料后可以重新开启再测。

报告包含 Recall@1/@3/@5/@10、Precision、MRR、NDCG，以及 mean/median/P95
检索延迟。CrossEncoder 首次运行会下载并缓存
`cross-encoder/ms-marco-MiniLM-L6-v2`；模型加载造成的冷启动会体现在 mean 中，
而 median 更接近预热后的单请求延迟。

以下环境变量控制检索，默认值也列在 `.env.example`：

```dotenv
RETRIEVAL_STRATEGY=hybrid
VECTOR_TOP_K=30
LEXICAL_TOP_K=30
RRF_K=60
RERANKER_ENABLED=true
RERANKER_MODEL_NAME=cross-encoder/ms-marco-MiniLM-L6-v2
RETRIEVAL_CANDIDATE_K=30
FINAL_TOP_K=5
NEIGHBOUR_EXPANSION_ENABLED=true
NEIGHBOUR_RADIUS=1
QUERY_REWRITE_ENABLED=false
QUERY_REWRITE_MODE=multi_query
MAX_QUERIES=3
```

`RETRIEVAL_STRATEGY` 支持 `vector_only`、`lexical_only`、`hybrid`；rewrite mode
支持 `single` 和 `multi_query`。改写失败会使用原问题，reranker 失败会使用 RRF
顺序，任一检索后端失败时会使用另一个后端。结果保留 `document_id`、`chunk_id`、
`retrieval_score`、`rerank_score`、`original_rank`、`final_rank` 和
`retrieval_source`，便于调试与离线评估。

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
