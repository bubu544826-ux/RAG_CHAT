"""Interface tests for the FastAPI application."""

import unittest
from unittest.mock import Mock

from fastapi.testclient import TestClient

from src.rag_app.api import app, get_rag_service


class ChatEndpointTest(unittest.TestCase):
    def setUp(self) -> None:
        self.rag_service = Mock()
        app.dependency_overrides[get_rag_service] = lambda: self.rag_service
        self.client = TestClient(app)

    def tearDown(self) -> None:
        app.dependency_overrides.clear()
        self.client.close()

    def test_root_serves_web_ui(self) -> None:
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers["content-type"])
        self.assertIn('id="question"', response.text)
        self.assertIn('id="send-button"', response.text)
        self.assertIn("Answer", response.text)
        self.assertIn("Sources", response.text)

    def test_web_ui_assets_are_served(self) -> None:
        script_response = self.client.get("/static/app.js")
        style_response = self.client.get("/static/styles.css")

        self.assertEqual(script_response.status_code, 200)
        self.assertIn('fetch("/chat"', script_response.text)
        self.assertIn("Loading", self.client.get("/").text)
        self.assertEqual(style_response.status_code, 200)
        self.assertIn("text/css", style_response.headers["content-type"])

    def test_chat_returns_answer_and_sources_from_rag_service(self) -> None:
        self.rag_service.ask.return_value = {
            "answer": "RAG 是检索增强生成。",
            "sources": [
                {
                    "source": "rag_notes.md",
                    "chunk_id": "rag_notes.md#chunk-0",
                    "score": 0.91,
                }
            ],
        }

        response = self.client.post("/chat", json={"question": "什么是 RAG？"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), self.rag_service.ask.return_value)
        self.rag_service.ask.assert_called_once_with("什么是 RAG？")

    def test_chat_rejects_an_empty_question(self) -> None:
        response = self.client.post("/chat", json={"question": ""})

        self.assertEqual(response.status_code, 422)
        self.rag_service.ask.assert_not_called()


if __name__ == "__main__":
    unittest.main()
