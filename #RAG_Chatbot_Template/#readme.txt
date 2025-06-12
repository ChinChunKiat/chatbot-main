#### Preliminary Setup

Run this to create the environment to initiate

1. create folder :
python -m venv testground

2. activate it:  !!! To run the program everytime, must activate this environment
testground\Scripts\activate

3. Install these all:
pip install chromadb
pip install langchain
pip install python-dotenv
pip install sentence_transformers
pip install flask
pip install -U langchain-community
pip install openai

to manage the vector database, install the #ManageChromaDB_DB.Browser.for.SQLite-v3.13.1-win64.msi in folder

4. Run the code by:
python app.py


***Optional If you want to go for testing, can run each individual file as their "Testing" Section that I mark with comments ###
document_processor.py: Show that it processed text files by splitting it
embedding_indexer.py: Show that it embed those text with Vectors
rag_chain.py: Show that it retrieve the information (Vectors) from database (vector database: Chroma)
chatbot.py: Run a chatbot interface on Terminal (replace the data/bako.txt with any file you want)
app.py: Run the chatbot on website (upload any file you want)

