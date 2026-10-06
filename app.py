"""
Flask web API for the chatbot
"""
import os
from flask import Flask, request, jsonify, Response
from dotenv import load_dotenv
from chatbot import Chatbot

load_dotenv()

app = Flask(__name__)

# Initialize chatbot
try:
    chatbot = Chatbot(provider_name=os.getenv("LLM_PROVIDER", "openai"))
except ValueError as e:
    print(f"Error initializing chatbot: {e}")
    chatbot = None


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "chatbot_initialized": chatbot is not None
    })


@app.route('/chat', methods=['POST'])
def chat():
    """Chat endpoint"""
    if chatbot is None:
        return jsonify({"error": "Chatbot not initialized"}), 500
    
    data = request.get_json()
    
    if not data or "message" not in data:
        return jsonify({"error": "Missing 'message' field"}), 400
    
    user_message = data.get("message")
    stream = data.get("stream", False)
    
    try:
        if stream:
            def generate():
                for chunk in chatbot.chat(user_message, stream=True):
                    yield chunk
            
            return Response(generate(), mimetype='text/event-stream')
        else:
            response = chatbot.chat(user_message)
            return jsonify({
                "message": response,
                "status": "success"
            })
    
    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500


@app.route('/history', methods=['GET'])
def get_history():
    """Get conversation history"""
    if chatbot is None:
        return jsonify({"error": "Chatbot not initialized"}), 500
    
    return jsonify({
        "history": chatbot.get_conversation_history(),
        "stats": chatbot.get_stats()
    })


@app.route('/clear', methods=['POST'])
def clear_history():
    """Clear conversation history"""
    if chatbot is None:
        return jsonify({"error": "Chatbot not initialized"}), 500
    
    chatbot.clear_history()
    return jsonify({"status": "success", "message": "History cleared"})


@app.route('/config', methods=['GET', 'POST'])
def config():
    """Get or update chatbot configuration"""
    if chatbot is None:
        return jsonify({"error": "Chatbot not initialized"}), 500
    
    if request.method == 'GET':
        return jsonify({
            "model_config": chatbot.model_config,
            "system_prompt": chatbot.system_prompt
        })
    
    elif request.method == 'POST':
        data = request.get_json()
        
        if "model_config" in data:
            chatbot.set_model_config(**data["model_config"])
        
        if "system_prompt" in data:
            chatbot.set_system_prompt(data["system_prompt"])
        
        return jsonify({"status": "success", "message": "Configuration updated"})


@app.route('/providers', methods=['GET'])
def get_providers():
    """Get available LLM providers"""
    return jsonify({
        "available_providers": ["openai", "anthropic", "google"],
        "current_provider": os.getenv("LLM_PROVIDER", "openai")
    })


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    port = int(os.getenv("FLASK_PORT", 5000))
    debug = os.getenv("DEBUG", "False").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
