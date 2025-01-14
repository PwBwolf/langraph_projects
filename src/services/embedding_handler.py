from openai import OpenAI
from typing import List

class EmbeddingHandler:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.openai_client = OpenAI()

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts using the selected model."""
        if self.model_name == "openai/text-embedding-3-small":
            return [self._emb_text_openai(text) for text in texts]
        else:
            raise ValueError(f"Unsupported embedding model: {self.model_name}")

    def _emb_text_openai(self, text: str) -> List[float]:
        """Generate an embedding using OpenAI's API."""
        response = self.openai_client.embeddings.create(
            input=text, model="text-embedding-3-small"
        )
        return response.data[0].embedding