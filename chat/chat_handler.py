"""
Chat Handler

Manages chat sessions and coordinates command parsing and execution.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from datetime import datetime


class ChatMessage:
    """Represents a chat message."""
    
    def __init__(self, role: str, content: str, timestamp: Optional[datetime] = None):
        self.role = role  # "user" or "assistant"
        self.content = content
        self.timestamp = timestamp or datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat()
        }


class ChatHandler:
    """
    Handles chat sessions for the Spine animation system.
    """
    
    def __init__(self, analyzer: Any, processor: Any):
        """
        Initialize with prompt analyzer and command processor.
        
        Args:
            analyzer: PromptAnalyzer instance
            processor: CommandProcessor instance
        """
        self.analyzer = analyzer
        self.processor = processor
        self.messages: List[ChatMessage] = []
        self.context: Dict[str, Any] = {}
    
    def add_user_message(self, content: str) -> ChatMessage:
        """
        Add a user message to the chat.
        
        Args:
            content: User message content
            
        Returns:
            ChatMessage instance
        """
        message = ChatMessage(role="user", content=content)
        self.messages.append(message)
        return message
    
    def add_assistant_message(self, content: str) -> ChatMessage:
        """
        Add an assistant message to the chat.
        
        Args:
            content: Assistant message content
            
        Returns:
            ChatMessage instance
        """
        message = ChatMessage(role="assistant", content=content)
        self.messages.append(message)
        return message
    
    def process_user_input(self, user_input: str) -> Dict[str, Any]:
        """
        Process user input and generate response.
        
        Args:
            user_input: User's message
            
        Returns:
            Response dict with message and action info
        """
        # Add user message
        self.add_user_message(user_input)
        
        # Analyze the prompt
        command = self.analyzer.analyze_prompt(user_input, self.context)
        
        # Process the command
        result = self.processor.process_command(command)
        
        # Add assistant response
        response_text = result.get("message", "I processed your request.")
        self.add_assistant_message(response_text)
        
        # Update context
        if result.get("success"):
            self._update_context(command, result)
        
        return {
            "response": response_text,
            "action": result.get("action"),
            "data": result,
            "command": command
        }
    
    def _update_context(self, command: Dict[str, Any], result: Dict[str, Any]):
        """
        Update chat context based on command and result.
        """
        intent = command.get("intent")
        params = command.get("parameters", {})
        
        if intent == "create_character":
            self.context["last_character"] = params.get("description")
        
        elif intent == "play_animation":
            self.context["current_animation"] = params.get("animation")
    
    def get_messages(self) -> List[Dict[str, Any]]:
        """
        Get all chat messages.
        
        Returns:
            List of message dicts
        """
        return [msg.to_dict() for msg in self.messages]
    
    def clear_history(self):
        """Clear chat history."""
        self.messages = []
        self.context = {}


if __name__ == "__main__":
    print("Chat handler module loaded successfully")

