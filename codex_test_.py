"""Create and visualize sentence embeddings as vectors in 3D.

The embedding model produces vectors with many dimensions.  The visualizer
uses PCA to project those vectors onto three dimensions, then draws each
sentence as an arrow from the origin.
"""

from collections.abc import Sequence

from scripts import visualize_embeddings


def main(arguments: Sequence[str] | None = None) -> int:
    """Run the 3D embedding visualization."""
    return visualize_embeddings.main(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
