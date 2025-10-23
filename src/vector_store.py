import os
import uuid
from typing import List, Tuple
import pinecone
from langchain.embeddings import OpenAIEmbeddings

class VectorStore:
    def __init__(self):
        """Initialize Pinecone vector store using the v2 SDK."""
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("PINECONE_API_KEY is not set")

        pinecone.init(api_key=api_key, environment=os.getenv("PINECONE_ENVIRONMENT", "us-east-1-aws"))

        # self.index_name = os.getenv("PINECONE_INDEX_NAME")
        self.index_name = "rag2"
        if not self.index_name:
            raise ValueError("PINECONE_INDEX_NAME is not set")

        # Explicitly set the OpenAI embeddings model and match dimension
        # text-embedding-ada-002 has dimension 1536
        self.embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
        self.embedding_dimension = 1536

        # Create index if it does not exist
        if self.index_name not in pinecone.list_indexes():
            pinecone.create_index(
                name=self.index_name,
                dimension=self.embedding_dimension,
                metric="cosine"
            )

        self.index = pinecone.Index(self.index_name)
    
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

        # Handle Pinecone v2 response format
        matches = results.get("matches", [])
        output: List[Tuple[str, float]] = []
        for m in matches:
            meta = m.get("metadata", {})
            score = m.get("score", 0.0)
            if meta and "text" in meta:
                output.append((meta["text"], float(score)))
        return output
    
    def as_retriever(self):
        """
        Return self as a retriever interface for LangChain
        """
        return self