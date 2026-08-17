"""Draw sentence embeddings as 3D vectors after reducing them with PCA."""

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from src.rag_app.embedding import EmbeddingError, embed_text


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_SENTENCES = [
    "猫喜欢在窗台上晒太阳。",
    "小狗在院子里追着球跑。",
    "这只鸟每天清晨都在唱歌。",
    "我用 Python 写了一个排序算法。",
    "这段 JavaScript 代码有一个空指针错误。",
    "编译器把源代码翻译成机器码。",
    "今天下午下了一场很大的雨。",
    "明天的气温会降到零度以下。",
]


def _parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="把句子的 embedding 向量降到 3 维并画出来。"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="每行一个句子的文本文件（默认：使用内置示例句子）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "embeddings_3d.png",
        help="图片保存路径",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="保存后再打开交互窗口，可以旋转查看",
    )
    return parser.parse_args(arguments)


def load_sentences(input_path: Path | None) -> list[str]:
    """Read one sentence per line, or fall back to the built-in examples."""
    if input_path is None:
        return list(DEFAULT_SENTENCES)

    lines = input_path.read_text(encoding="utf-8").splitlines()
    sentences = [line.strip() for line in lines if line.strip()]
    if len(sentences) < 3:
        raise ValueError("至少需要 3 个句子才能降到 3 维。")
    return sentences


def embed_sentences(sentences: Sequence[str]) -> np.ndarray:
    """Turn every sentence into a vector and stack them into one matrix."""
    vectors = [embed_text(sentence) for sentence in sentences]
    return np.array(vectors, dtype=float)


def reduce_to_3d(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project high-dimensional vectors onto their 3 main directions with PCA.

    PCA 先把数据移到原点，再用 SVD 找出方差最大的三个方向，
    最后把每个向量投影到这三个方向上，得到可以画出来的 (x, y, z)。
    """
    centered = vectors - vectors.mean(axis=0)
    _, singular_values, components = np.linalg.svd(centered, full_matrices=False)

    coordinates = centered @ components[:3].T

    total_variance = np.sum(singular_values**2)
    explained_ratio = singular_values[:3] ** 2 / total_variance
    return coordinates, explained_ratio


def plot_vectors(
    coordinates: np.ndarray,
    sentences: Sequence[str],
    explained_ratio: np.ndarray,
    output_path: Path,
    show: bool,
) -> None:
    """Draw one arrow per sentence from the origin to its 3D coordinate."""
    import matplotlib

    if not show:
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    # 中文标签需要一个能显示汉字的字体，否则 matplotlib 会画成方块。
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False

    figure = plt.figure(figsize=(11, 9))
    axes = figure.add_subplot(projection="3d")

    colors = plt.get_cmap("tab10")(np.linspace(0, 1, len(sentences)))

    for index, (point, sentence) in enumerate(zip(coordinates, sentences)):
        x, y, z = point
        axes.quiver(0, 0, 0, x, y, z, color=colors[index], arrow_length_ratio=0.12)
        axes.scatter(x, y, z, color=colors[index], s=40)
        axes.text(x, y, z, f"  {index + 1}", fontsize=9)

    axes.set_xlabel(f"PC1 ({explained_ratio[0]:.1%})")
    axes.set_ylabel(f"PC2 ({explained_ratio[1]:.1%})")
    axes.set_zlabel(f"PC3 ({explained_ratio[2]:.1%})")
    axes.set_title(f"{len(sentences)} 个句子的 embedding 向量（PCA 降到 3 维）")

    legend_text = "\n".join(
        f"{index + 1}. {sentence}" for index, sentence in enumerate(sentences)
    )
    figure.text(0.02, 0.02, legend_text, fontsize=8, va="bottom")
    figure.subplots_adjust(bottom=0.05 + 0.022 * len(sentences))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    print(f"图片已保存到：{output_path}")

    if show:
        plt.show()
    plt.close(figure)


def main(arguments: Sequence[str] | None = None) -> int:
    """Embed the sentences, reduce them to 3D, and save the plot."""
    args = _parse_arguments(arguments)

    try:
        sentences = load_sentences(args.input)
    except (OSError, ValueError) as exc:
        print(f"读取句子失败：{exc}")
        return 1

    try:
        vectors = embed_sentences(sentences)
    except (TypeError, ValueError, EmbeddingError) as exc:
        print(f"生成 embedding 失败：{exc}")
        return 1

    print(f"句子数量：{len(sentences)}")
    print(f"原始向量维度：{vectors.shape[1]}")

    coordinates, explained_ratio = reduce_to_3d(vectors)
    print(f"前 3 个主成分保留的信息比例：{explained_ratio.sum():.1%}")

    try:
        plot_vectors(coordinates, sentences, explained_ratio, args.output, args.show)
    except ImportError:
        print("缺少 matplotlib，请先安装 requirements.txt 中的依赖。")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
