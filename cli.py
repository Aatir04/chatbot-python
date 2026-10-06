"""
Command-line interface for the chatbot
"""
import os
import sys
from dotenv import load_dotenv
from chatbot import Chatbot

load_dotenv()


def print_welcome():
    """Print welcome message"""
    print("\n" + "="*60)
    print("🤖 Multi-LLM Chatbot")
    print("="*60)
    print("Commands:")
    print("  /quit or /exit - Exit the chatbot")
    print("  /clear - Clear conversation history")
    print("  /stats - Show conversation statistics")
    print("  /provider <name> - Switch LLM provider (openai, anthropic, google)")
    print("  /history - Show conversation history")
    print("="*60 + "\n")


def handle_command(command: str, chatbot: Chatbot) -> bool:
    """
    Handle special commands
    
    Returns:
        True if the chatbot should continue, False if it should exit
    """
    parts = command.strip().split(maxsplit=1)
    cmd = parts[0].lower()
    
    if cmd in ["/quit", "/exit"]:
        print("Goodbye! 👋")
        return False
    
    elif cmd == "/clear":
        chatbot.clear_history()
        print("✓ Conversation history cleared")
        return True
    
    elif cmd == "/stats":
        stats = chatbot.get_stats()
        print(f"\nConversation Statistics:")
        print(f"  Messages: {stats['message_count']}")
        print(f"  Created: {stats['created_at']}")
        print(f"  Duration: {stats['duration']:.1f}s\n")
        return True
    
    elif cmd == "/history":
        history = chatbot.get_conversation_history()
        if not history:
            print("No conversation history yet.\n")
        else:
            print("\nConversation History:")
            for i, msg in enumerate(history, 1):
                print(f"\n{i}. {msg['role'].upper()}:")
                print(f"   {msg['content'][:100]}{'...' if len(msg['content']) > 100 else ''}")
            print()
        return True
    
    elif cmd == "/provider":
        if len(parts) < 2:
            print("Usage: /provider <name> (openai, anthropic, google)")
            return True
        
        provider_name = parts[1].lower()
        try:
            chatbot.provider = __import__('chatbot').LLMFactory.create_provider(provider_name)
            print(f"✓ Switched to {provider_name} provider\n")
        except ValueError as e:
            print(f"✗ Error: {e}\n")
        return True
    
    else:
        print(f"Unknown command: {cmd}")
        return True


def main():
    """Main CLI loop"""
    print_welcome()
    
    # Get provider from environment or use default
    provider = os.getenv("LLM_PROVIDER", "openai")
    
    try:
        chatbot = Chatbot(provider_name=provider)
        print(f"✓ Chatbot initialized with {provider} provider\n")
    except ValueError as e:
        print(f"✗ Error initializing chatbot: {e}")
        print("Please set the appropriate API keys in your .env file")
        sys.exit(1)
    
    # Main conversation loop
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.startswith("/"):
                if not handle_command(user_input, chatbot):
                    break
                continue
            
            # Get response from chatbot
            print("\nBot: ", end="", flush=True)
            response = chatbot.chat(user_input, stream=False)
            print(response)
            print()
        
        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"\n✗ Error: {e}\n")


if __name__ == "__main__":
    main()
