"""
Main Chatbot class with conversation management
"""
from typing import List, Dict, Optional
from datetime import datetime
from .llm_provider import LLMFactory, LLMProvider


class ConversationMemory:
    """Manages conversation history"""
    
    def __init__(self, max_history: int = 10):
        self.messages: List[Dict[str, str]] = []
        self.max_history = max_history
        self.created_at = datetime.now()
    
    def add_message(self, role: str, content: str):
        """Add a message to the conversation history"""
        self.messages.append({
            "role": role,
            "content": content
        })
        # Keep only the last max_history messages
        if len(self.messages) > self.max_history * 2:
            self.messages = self.messages[-self.max_history*2:]
    
    def get_messages(self) -> List[Dict[str, str]]:
        """Get the full conversation history"""
        return self.messages
    
    def clear(self):
        """Clear conversation history"""
        self.messages = []
    
    def get_summary(self) -> Dict:
        """Get conversation summary"""
        return {
            "message_count": len(self.messages),
            "created_at": self.created_at.isoformat(),
            "duration": (datetime.now() - self.created_at).total_seconds()
        }


class Chatbot:
    """Main Chatbot class"""
    
    def __init__(self, provider_name: Optional[str] = None, system_prompt: Optional[str] = None):
        """
        Initialize the chatbot
        
        Args:
            provider_name: LLM provider to use (openai, anthropic, google)
            system_prompt: Custom system prompt for the chatbot
        """
        self.provider: LLMProvider = LLMFactory.get_provider(provider_name)
        self.memory = ConversationMemory()
        self.system_prompt = system_prompt or self._get_default_system_prompt()
        self.model_config = {
            "temperature": 0.7,
            "max_tokens": 500
        }
    
    @staticmethod
    def _get_default_system_prompt() -> str:
        """Get the default system prompt"""
        return """You are a helpful, friendly, and knowledgeable AI assistant. 
You provide clear, concise, and accurate responses. 
You ask clarifying questions when needed and maintain a conversational tone."""
    
    def set_system_prompt(self, prompt: str):
        """Update the system prompt"""
        self.system_prompt = prompt
    
    def set_model_config(self, **kwargs):
        """Update model configuration"""
        self.model_config.update(kwargs)
    
    def chat(self, user_message: str, stream: bool = False):
        """
        Send a message to the chatbot
        
        Args:
            user_message: The user's message
            stream: Whether to stream the response
        
        Returns:
            The chatbot's response (or generator if streaming)
        """
        # Add user message to memory
        self.memory.add_message("user", user_message)
        
        # Prepare messages with system prompt
        messages = [{"role": "system", "content": self.system_prompt}] + self.memory.get_messages()
        
        if stream:
            # Stream the response
            response_text = ""
            for chunk in self.provider.stream_chat(messages, **self.model_config):
                response_text += chunk
                yield chunk
            # Add the complete response to memory
            self.memory.add_message("assistant", response_text)
        else:
            # Get the full response
            response = self.provider.chat(messages, **self.model_config)
            # Add response to memory
            self.memory.add_message("assistant", response)
            return response
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get the current conversation history"""
        return self.memory.get_messages()
    
    def clear_history(self):
        """Clear the conversation history"""
        self.memory.clear()
    
    def get_stats(self) -> Dict:
        """Get conversation statistics"""
        return self.memory.get_summary()
