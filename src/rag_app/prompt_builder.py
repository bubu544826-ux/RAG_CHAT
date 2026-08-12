"""Build a prompt from a question and retrieved knowledge chunks."""


SYSTEM_INSTRUCTION = """你是一个基于知识库回答问题的助手。
请遵循以下规则：
1. 优先根据提供的 context 回答问题。
2. 如果 context 中没有足够信息回答问题，请明确说明“根据提供的 context 无法回答该问题”。
3. 不要编造 context 中不存在的信息。"""


def build_prompt(question: str, retrieved_chunks: list[dict[str, object]]) -> str:
    """Return one prompt containing instructions, context, and the question."""
    if not isinstance(question, str):
        raise TypeError("question 必须是字符串。")
    if not question.strip():
        raise ValueError("question 不能为空。")
    if not isinstance(retrieved_chunks, list):
        raise TypeError("retrieved_chunks 必须是列表。")

    context_parts: list[str] = []

    for position, chunk in enumerate(retrieved_chunks, start=1):
        if not isinstance(chunk, dict):
            raise TypeError("retrieved_chunks 中的每个 chunk 必须是字典。")

        text = chunk.get("text")
        if not isinstance(text, str):
            raise ValueError("每个 chunk 必须包含字符串类型的 text 字段。")

        source = chunk.get("source", "未知来源")
        chunk_id = chunk.get("chunk_id", "未知编号")
        context_parts.append(
            f"[Chunk {position}]\n"
            f"来源：{source}\n"
            f"编号：{chunk_id}\n"
            f"内容：{text}"
        )

    context = (
        "\n\n".join(context_parts)
        if context_parts
        else "（没有检索到相关内容）"
    )

    return (
        f"{SYSTEM_INSTRUCTION}\n\n"
        f"context:\n{context}\n\n"
        f"question:\n{question}"
    )
