from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

class RAGChain:
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
        self.llm = self.get_llm()

    # A function: Initiliase one of the language model
    def get_llm(self):
        if os.getenv("DEEPSEEK_API_KEY"): # api key is in the .env file
            return ChatOpenAI(
                openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
                base_url="https://openrouter.ai/api/v1",   # OpenRouter point
                model="deepseek/deepseek-chat-v3-0324:free",  # deepseek model
                temperature=0,
            )
        elif os.getenv("OPENAI_API_KEY"):
            return OpenAI(api_key=os.getenv("OPENAI_API_KEY"), temperature=0)
        elif os.getenv("GEMINI_API_KEY"):
            return Gemini(api_key=os.getenv("GEMINI_API_KEY"), temperature=0)
        elif os.getenv("FIREWORKS_API_KEY"):
            return Fireworks(api_key=os.getenv("FIREWORKS_API_KEY"), temperature=0)
        else:
            raise ValueError("No valid API key found! Please set one in .env file.")

    def create_chain(self):
        # Convert the vector in a retrieval object
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3}) # Get top 3 most relevant object 
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff", # Merges all retrieved documents into a single LLM prompt
            retriever=retriever, # Link the retriever from above to get relevant documents
            return_source_documents=True # Includes retrieved documents in the output
        )
        return qa_chain # Make it a callable object that can answer questions using RAG.



################ Testing ################
if __name__ == "__main__":
    from document_processor import DocumentProcessor
    from embedding_indexer import EmbeddingIndexer

    processor = DocumentProcessor("data/bako.txt")
    texts = processor.load_and_split()

    indexer = EmbeddingIndexer()
    vectorstore = indexer.create_vectorstore(texts)

    rag_chain = RAGChain(vectorstore)
    qa_chain = rag_chain.create_chain()

    query = "where do i celebrate Gawai Dayak 2025 in Betong, Sarawak?"
    result = qa_chain({"query": query})
    print(f"Answer: {result['result']}")