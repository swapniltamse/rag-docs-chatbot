import os
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from rag_engine import RAGEngine

# Load environment variables
load_dotenv()

app = Flask(__name__)
rag_engine = RAGEngine(
    docs_dir="static_docs",
    db_dir="chroma_db"
)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    query = data.get('query', '')
    
    if not query:
        return jsonify({"error": "Query is required"}), 400
    
    # Get response using RAG approach
    response_data = rag_engine.generate_response(query)
    
    # Check if response is a dictionary (new format) or string (old format)
    if isinstance(response_data, dict):
        response_text = response_data.get("answer", "")
    else:
        response_text = response_data
    
    return jsonify({
        "query": query,
        "response": response_text
    })
@app.route('/api/reload', methods=['POST'])
def reload_docs():
    """Endpoint to reload and reindex documents"""
    try:
        rag_engine.load_documents()
        return jsonify({"success": "Documents reloaded successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Make sure docs are loaded
    if not rag_engine.is_db_initialized():
        rag_engine.load_documents()
    
    # Start app
    app.run(debug=True)