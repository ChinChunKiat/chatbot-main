from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma  # Replace FAISS with Chroma
from langchain.docstore.document import Document

class EmbeddingIndexer:
    def __init__(self, persist_directory="./chroma_db"):
        # Initialize embeddings (same as before)
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        # Directory to persist the Chroma database
        self.persist_directory = persist_directory

    def create_vectorstore(self, texts):
        """
        Creates a Chroma vectorstore from a list of texts.
        Args:
            texts: List of LangChain `Document` objects or raw texts.
        Returns:
            Chroma vectorstore instance.
        """
        # Ensure texts are in Document format (if not already)
        if isinstance(texts[0], str):
            texts = [Document(page_content=text) for text in texts]

        # Create Chroma vectorstore (persists to disk automatically if `persist_directory` is set)
        vectorstore = Chroma.from_documents(
            documents=texts,
            embedding=self.embeddings,
            persist_directory=self.persist_directory  # Saves to disk
        )
        return vectorstore

    # Load an existing Chroma vectorstore
    def load_vectorstore(self):
        return Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        )

################ Testing ################
if __name__ == "__main__":
    from document_processor import DocumentProcessor

    processor = DocumentProcessor("data/sample.txt")
    texts = processor.load_and_split()

    indexer = EmbeddingIndexer() # Create an embedding instance
    vectorstore = indexer.create_vectorstore(texts) # Embed the splitted 'texts' then put into vector store
    print("Vector store created successfully")