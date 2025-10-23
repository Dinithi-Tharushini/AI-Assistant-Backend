from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tempfile
import os

class DocumentProcessor:
    def __init__(self, vector_store):
        """
        Initialize document processor
        """
        self.vector_store = vector_store
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1100,
            chunk_overlap=200,
            length_function=len
        )
    
    def process_document(self, file):
        """
        Process uploaded document and store in vector store.
        Ensures temp file handles are closed before deletion (fixes WinError 32 on Windows).
        """
        tmp_path = None
        try:
            # Save upload to a temp file, then close handle before reading
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp_path = tmp.name
                file.save(tmp_path)

            # Load document based on file type
            if file.filename.lower().endswith('.pdf'):
                loader = PyPDFLoader(tmp_path)
            else:
                loader = TextLoader(tmp_path, autodetect_encoding=True)

            documents = loader.load()

            # Split text into chunks
            texts = self.text_splitter.split_documents(documents)

            # Extract text content and metadata
            text_contents = [doc.page_content for doc in texts]
            metadata = [doc.metadata for doc in texts]

            # Add to vector store
            self.vector_store.add_texts(text_contents, metadata)
        finally:
            # Clean up temporary file after loaders have released it
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except PermissionError:
                    pass
