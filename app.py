from flask import Flask, render_template, request, jsonify
from chatbot_logic import get_response

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.form['message']
    bot_type = request.form['bot_type']
    response = get_response(user_input, bot_type)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
