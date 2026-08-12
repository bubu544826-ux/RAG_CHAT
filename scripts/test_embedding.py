"""Interactively inspect an embedding without printing the full vector."""

from src.rag_app.embedding import EmbeddingError, embed_text


def main() -> int:
    text = input("请输入一句话：")

    try:
        vector = embed_text(text)
    except (TypeError, ValueError, EmbeddingError) as exc:
        print(f"生成 embedding 失败：{exc}")
        return 1

    print(f"vector 类型：{type(vector).__name__}")
    print(f"vector 维度：{len(vector)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
