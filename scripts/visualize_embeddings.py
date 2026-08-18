"""Draw sentence embeddings as 3D vectors after reducing them with PCA."""

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from src.rag_app.embedding import EmbeddingError, embed_text


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_SENTENCES = [
    "The cat likes to sunbathe on the windowsill.",
    "The puppy chased a ball around the yard.",
    "This bird sings every morning at dawn.",
    "I wrote a sorting algorithm in Python.",
    "This JavaScript code has a null pointer bug.",
    "A compiler translates source code into machine code.",
    "It rained heavily this afternoon.",
    "Tomorrow the temperature will drop below zero.",
]


def _parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reduce sentence embedding vectors to 3 dimensions and plot them."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="text file with one sentence per line (default: use the built-in example sentences)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "embeddings_3d.png",
        help="path to save the image to",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="after saving, open an interactive window so the plot can be rotated",
    )
    return parser.parse_args(arguments)


def load_sentences(input_path: Path | None) -> list[str]:
    """Read one sentence per line, or fall back to the built-in examples."""
    if input_path is None:
        return list(DEFAULT_SENTENCES)

    lines = input_path.read_text(encoding="utf-8").splitlines()
    sentences = [line.strip() for line in lines if line.strip()]
    if len(sentences) < 3:
        raise ValueError("At least 3 sentences are needed to reduce to 3 dimensions.")
    return sentences


def embed_sentences(sentences: Sequence[str]) -> np.ndarray:
    """Turn every sentence into a vector and stack them into one matrix."""
    vectors = [embed_text(sentence) for sentence in sentences]
    return np.array(vectors, dtype=float)


def reduce_to_3d(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project high-dimensional vectors onto their 3 main directions with PCA.

    PCA first moves the data to the origin, then uses SVD to find the three
    directions of greatest variance, and finally projects every vector onto
    those directions to get the plottable (x, y, z).
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

    matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans"]
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
    axes.set_title(f"Embedding vectors of {len(sentences)} sentences (PCA down to 3D)")

    legend_text = "\n".join(
        f"{index + 1}. {sentence}" for index, sentence in enumerate(sentences)
    )
    figure.text(0.02, 0.02, legend_text, fontsize=8, va="bottom")
    figure.subplots_adjust(bottom=0.05 + 0.022 * len(sentences))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    print(f"Image saved to: {output_path}")

    if show:
        plt.show()
    plt.close(figure)


def main(arguments: Sequence[str] | None = None) -> int:
    """Embed the sentences, reduce them to 3D, and save the plot."""
    args = _parse_arguments(arguments)

    try:
        sentences = load_sentences(args.input)
    except (OSError, ValueError) as exc:
        print(f"Failed to read the sentences: {exc}")
        return 1

    try:
        vectors = embed_sentences(sentences)
    except (TypeError, ValueError, EmbeddingError) as exc:
        print(f"Failed to generate the embeddings: {exc}")
        return 1

    print(f"sentences: {len(sentences)}")
    print(f"original vector dimension: {vectors.shape[1]}")

    coordinates, explained_ratio = reduce_to_3d(vectors)
    print(f"variance kept by the first 3 principal components: {explained_ratio.sum():.1%}")

    try:
        plot_vectors(coordinates, sentences, explained_ratio, args.output, args.show)
    except ImportError:
        print("matplotlib is missing; install the dependencies listed in requirements.txt first.")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
