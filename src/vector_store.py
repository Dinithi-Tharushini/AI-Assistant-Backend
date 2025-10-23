import os
import uuid
from typing import List, Tuple
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings

class VectorStore:
    def __init__(self):
        """Initialize Pinecone vector store using the v3 SDK."""
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("PINECONE_API_KEY is not set")

        self.pc = Pinecone(api_key=api_key)

        # self.index_name = os.getenv("PINECONE_INDEX_NAME")
        self.index_name = "rag2"
        if not self.index_name:
            raise ValueError("PINECONE_INDEX_NAME is not set")

        # Explicitly set the OpenAI embeddings model and match dimension
        # text-embedding-3-small has dimension 1536
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.embedding_dimension = 1536

        # Serverless index config (can be overridden via env)
        cloud = os.getenv("PINECONE_CLOUD", "aws")
        region = os.getenv("PINECONE_REGION", "us-east-1")

        # Create index if it does not exist
        existing = self.pc.list_indexes().names()
        if self.index_name not in existing:
            self.pc.create_index(
                name=self.index_name,
                dimension=self.embedding_dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud=cloud, region=region),
            )

        self.index = self.pc.Index(self.index_name)
    
    def add_texts(self, texts: List[str], metadata: List[dict] | None = None) -> None:
        """Add texts to the vector store with generated unique IDs."""
        if not texts:
            return

        embeddings = self.embeddings.embed_documents(texts)

        vectors = []
        for i, (text, embedding) in enumerate(zip(texts, embeddings)):
            base_meta = metadata[i] if metadata and i < len(metadata) else {}
            # Ensure the retriever can return text directly from metadata
            safe_meta = {**base_meta, "text": text}
            vectors.append({
                "id": str(uuid.uuid4()),
                "values": embedding,
                "metadata": safe_meta,
            })

        # Upsert in small batches to avoid Pinecone 4MB request limit
        batch_size = 20
        for start in range(0, len(vectors), batch_size):
            batch = vectors[start:start + batch_size]
            if batch:
                self.index.upsert(vectors=batch)
    
    def similarity_search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """Perform similarity search and return list of (text, score)."""
        query_embedding = self.embeddings.embed_query(query)
        results = self.index.query(
            vector=query_embedding,
            top_k=k,
            include_metadata=True,
        )

        # Support both dict-like and attribute access
        matches = results.get("matches", []) if isinstance(results, dict) else getattr(results, "matches", [])
        output: List[Tuple[str, float]] = []
        for m in matches:
            meta = m.get("metadata") if isinstance(m, dict) else getattr(m, "metadata", {})
            score = m.get("score") if isinstance(m, dict) else getattr(m, "score", 0.0)
            if meta and "text" in meta:
                output.append((meta["text"], float(score)))
        return output
    
    def as_retriever(self):
        """
        Return self as a retriever interface for LangChain
        """
        return self