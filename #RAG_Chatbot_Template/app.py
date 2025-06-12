from flask import Flask, render_template, request, jsonify
from document_processor import DocumentProcessor
from embedding_indexer import EmbeddingIndexer
from rag_chain import RAGChain
from chatbot import Chatbot
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variable to store the chatbot instance
current_chatbot = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global current_chatbot
    
    try:
        # 1. File upload verification
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # 2. Save file (you confirmed this works)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'knowledge_base.txt')
        file.save(filepath)
        print(f"File saved to: {filepath}")  # Verify in console

        # 3. Process the document with error handling
        try:
            processor = DocumentProcessor(filepath)
            texts = processor.load_and_split()
            print(f"Processed {len(texts)} text chunks")  # Debug output
        except Exception as e:
            print(f"Document processing failed: {str(e)}")
            return jsonify({'error': f'Document processing failed: {str(e)}'}), 500

        # 4. Create vectorstore with error handling
        try:
            indexer = EmbeddingIndexer()
            vectorstore = indexer.create_vectorstore(texts)
            print("Vectorstore created successfully")  # Debug output
        except Exception as e:
            print(f"Vectorstore creation failed: {str(e)}")
            return jsonify({'error': f'Vectorstore creation failed: {str(e)}'}), 500

        # 5. Initialize chatbot
        try:
            rag_chain = RAGChain(vectorstore)
            current_chatbot = Chatbot(rag_chain.create_chain())
            print("Chatbot initialized successfully")  # Debug output
        except Exception as e:
            print(f"Chatbot initialization failed: {str(e)}")
            return jsonify({'error': f'Chatbot initialization failed: {str(e)}'}), 500

        return jsonify({'success': True, 'message': 'File processed successfully!'})

    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return jsonify({'error': f'Unexpected server error: {str(e)}'}), 500

@app.route('/chat', methods=['POST'])
def chat():
    global current_chatbot
    
    if not current_chatbot:
        return jsonify({'error': 'Please upload a knowledge base first'}), 400
    
    data = request.get_json()
    prompt = data.get('prompt')
    
    if not prompt:
        return jsonify({'error': 'No prompt provided'}), 400
    
    response = current_chatbot.get_response(prompt)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)