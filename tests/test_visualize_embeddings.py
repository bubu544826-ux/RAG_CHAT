"""Tests for the 3D embedding visualization script."""

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts.visualize_embeddings import load_sentences, main, reduce_to_3d


def _fake_embed_text(text: str) -> list[float]:
    """Return a deterministic 8-dimensional vector so tests need no model."""
    seed = sum(ord(character) for character in text)
    generator = np.random.default_rng(seed)
    return generator.normal(size=8).tolist()


class LoadSentencesTest(unittest.TestCase):
    def test_uses_builtin_sentences_when_no_file_given(self) -> None:
        sentences = load_sentences(None)

        self.assertGreaterEqual(len(sentences), 3)

    def test_reads_one_sentence_per_line_and_skips_blanks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "sentences.txt"
            input_path.write_text("第一句\n\n第二句\n  第三句  \n", encoding="utf-8")

            sentences = load_sentences(input_path)

        self.assertEqual(sentences, ["第一句", "第二句", "第三句"])

    def test_rejects_fewer_than_three_sentences(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "sentences.txt"
            input_path.write_text("只有一句\n", encoding="utf-8")

            with self.assertRaises(ValueError):
                load_sentences(input_path)


class ReduceTo3dTest(unittest.TestCase):
    def test_returns_three_coordinates_per_vector(self) -> None:
        vectors = np.random.default_rng(0).normal(size=(6, 16))

        coordinates, explained_ratio = reduce_to_3d(vectors)

        self.assertEqual(coordinates.shape, (6, 3))
        self.assertEqual(explained_ratio.shape, (3,))

    def test_keeps_all_information_when_data_is_already_3d(self) -> None:
        # 四个点最多张成 3 维空间，所以前 3 个主成分应该保留全部信息。
        vectors = np.random.default_rng(1).normal(size=(4, 16))

        _, explained_ratio = reduce_to_3d(vectors)

        self.assertAlmostEqual(float(explained_ratio.sum()), 1.0, places=6)


class MainTest(unittest.TestCase):
    @patch("scripts.visualize_embeddings.embed_text", side_effect=_fake_embed_text)
    def test_writes_image_file(self, embed_text_mock) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "plots" / "embeddings_3d.png"
            output = io.StringIO()

            with redirect_stdout(output):
                exit_code = main(["--output", str(output_path)])

            self.assertEqual(exit_code, 0)
            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 0)

        self.assertGreaterEqual(embed_text_mock.call_count, 3)
        self.assertIn("原始向量维度：8", output.getvalue())

    @patch(
        "scripts.visualize_embeddings.embed_text",
        side_effect=ValueError("text 不能为空。"),
    )
    def test_reports_embedding_failure(self, embed_text_mock) -> None:
        output = io.StringIO()

        with redirect_stdout(output):
            exit_code = main([])

        self.assertEqual(exit_code, 1)
        self.assertIn("生成 embedding 失败", output.getvalue())


if __name__ == "__main__":
    unittest.main()
