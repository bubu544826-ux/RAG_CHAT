# RAG 项目

这是项目第一阶段的工程骨架。目前只提供可运行的 Python 入口和烟雾测试，尚未实现任何 RAG、向量检索或模型调用逻辑。

## 项目结构

```text
.
├── src/
│   └── rag_app/          # 应用源代码包
│       ├── __init__.py
│       ├── __main__.py   # `python -m src.rag_app` 的模块入口
│       └── app.py        # 最小启动函数
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

当前阶段没有第三方运行时依赖，所以不创建虚拟环境也可以直接启动：

```powershell
python -m src.rag_app
```

## 运行测试

使用 `uv`：

```powershell
uv run --no-project --python 3.12 python -m unittest discover -s tests -v
```

或使用已安装并激活的 Python：

```powershell
python -m unittest discover -s tests -v
```
