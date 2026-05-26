import sqlite3
from flask import Flask, request, jsonify
from transformers import pipeline
import nltk
from datetime import datetime

# Download NLTK data required for tokenization
nltk.download('punkt', quiet=True)

app = Flask(__name__)

# Initialize the Hugging Face conversational pipeline
# Using DialoGPT-medium for intelligent contextual responses
print("Loading Transformer model... This may take a moment.")
chatbot = pipeline("text-generation", model="microsoft/DialoGPT-medium")

def init_db():
    """Initialize SQLite database for storing interaction logs."""
    conn = sqlite3.connect('chat_logs.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            user_input TEXT,
            bot_response TEXT
        )
    ''')
    conn.commit()
    conn.close()

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get("message", "")
    
    if not user_message:
        return jsonify({"error": "Message is required"}), 400

    # 1. NLTK Processing (Tokenizing input for NLP analysis)
    tokens = nltk.word_tokenize(user_message)
    
    # 2. Transformer Processing (Generating response)
    # Appending a bot tag to prompt the model to reply
    prompt = f"User: {user_message}\nBot:"
    response = chatbot(prompt, max_length=100, pad_token_id=50256, num_return_sequences=1)
    
    # Clean up the generated text to extract only the bot's reply
    raw_output = response[0]['generated_text']
    bot_reply = raw_output.split("Bot:")[-1].strip()

    # 3. SQLite Database Logging
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = sqlite3.connect('chat_logs.db')
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO interactions (timestamp, user_input, bot_response) VALUES (?, ?, ?)",
        (timestamp, user_message, bot_reply)
    )
    conn.commit()
    conn.close()

    return jsonify({
        "timestamp": timestamp,
        "user_input": user_message,
        "tokens": tokens,
        "bot_response": bot_reply
    })

if __name__ == '__main__':
    init_db()
    # Run the Flask app on localhost:5000
    app.run(debug=True, port=5000)