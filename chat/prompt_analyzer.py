"""
Prompt Analyzer

Uses AI to parse natural language commands into structured actions.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from openai import OpenAI


class PromptAnalyzer:
    """
    Analyzes user prompts and extracts structured commands.
    """
    
    def __init__(self, openai_client: Optional[OpenAI] = None):
        """
        Initialize with OpenAI client.
        
        Args:
            openai_client: OpenAI client instance (optional)
        """
        self.client = openai_client
    
    def analyze_prompt(self, user_message: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze a user prompt and extract intent and parameters.
        
        Args:
            user_message: User's natural language command
            context: Optional context about current state
            
        Returns:
            Dict with intent, action, and parameters
        """
        if not self.client:
            # Fallback to simple keyword matching
            return self._simple_analysis(user_message)
        
        try:
            # Use OpenAI to parse the command
            system_prompt = """You are a command parser for a Spine2D animation system. 
Parse user commands into structured JSON actions.

Available action types:
- "create_character": Generate a new character from description
- "add_animation": Add an animation to existing character
- "modify_animation": Edit an existing animation
- "modify_skeleton": Adjust skeleton/bones
- "export": Export the project
- "play_animation": Play an animation for preview

Return JSON with:
{
  "intent": "<action_type>",
  "parameters": {<extracted parameters>},
  "description": "<natural language summary of what will happen>"
}

Examples:
User: "Create a blue cat walking"
Response: {"intent": "create_character", "parameters": {"description": "blue cat", "initial_animation": "walk"}, "description": "Creating a blue cat character with walking animation"}

User: "Make the jump faster"
Response: {"intent": "modify_animation", "parameters": {"animation": "jump", "modification": "speed", "factor": 1.5}, "description": "Making the jump animation 1.5x faster"}

User: "Rotate the ankle at time 4 seconds by 15 degrees"
Response: {"intent": "modify_animation", "parameters": {"bone": "ankle", "time": 4.0, "property": "rotation", "value": 15}, "description": "Rotating ankle bone by 15 degrees at 4 seconds"}

User: "Make the legs longer"
Response: {"intent": "modify_skeleton", "parameters": {"limb": "leg", "modification": "length", "factor": 1.2}, "description": "Making legs 20% longer"}"""

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                response_format={"type": "json_object"},
                temperature=0.3
            )
            
            import json
            result = json.loads(response.choices[0].message.content)
            return result
        
        except Exception as e:
            print(f"Error analyzing prompt with OpenAI: {e}")
            return self._simple_analysis(user_message)
    
    def _simple_analysis(self, message: str) -> Dict[str, Any]:
        """
        Simple keyword-based analysis as fallback.
        """
        message_lower = message.lower()
        
        # Create character
        if any(word in message_lower for word in ["create", "generate", "make a"]):
            return {
                "intent": "create_character",
                "parameters": {"description": message},
                "description": f"Creating character: {message}"
            }
        
        # Add animation
        if "add" in message_lower and any(anim in message_lower for anim in ["walk", "jump", "wave", "idle"]):
            anim_name = None
            for anim in ["walk", "jump", "wave", "idle"]:
                if anim in message_lower:
                    anim_name = anim
                    break
            return {
                "intent": "add_animation",
                "parameters": {"animation": anim_name},
                "description": f"Adding {anim_name} animation"
            }
        
        # Modify animation speed
        if any(word in message_lower for word in ["faster", "slower", "speed"]):
            anim_name = None
            for anim in ["walk", "jump", "wave", "idle"]:
                if anim in message_lower:
                    anim_name = anim
                    break
            
            factor = 1.5 if "faster" in message_lower else 0.7
            
            return {
                "intent": "modify_animation",
                "parameters": {
                    "animation": anim_name,
                    "modification": "speed",
                    "factor": factor
                },
                "description": f"Adjusting {anim_name} animation speed"
            }
        
        # Modify skeleton
        if any(word in message_lower for word in ["longer", "shorter", "bigger", "smaller"]):
            return {
                "intent": "modify_skeleton",
                "parameters": {"description": message},
                "description": f"Modifying skeleton: {message}"
            }
        
        # Export
        if "export" in message_lower or "download" in message_lower:
            return {
                "intent": "export",
                "parameters": {},
                "description": "Exporting Spine project"
            }
        
        # Play animation
        for anim in ["walk", "jump", "wave", "idle"]:
            if anim in message_lower and ("play" in message_lower or "show" in message_lower):
                return {
                    "intent": "play_animation",
                    "parameters": {"animation": anim},
                    "description": f"Playing {anim} animation"
                }
        
        # Default: unclear intent
        return {
            "intent": "clarify",
            "parameters": {},
            "description": "I'm not sure what you want to do. Could you rephrase?"
        }


if __name__ == "__main__":
    # Test without OpenAI
    analyzer = PromptAnalyzer()
    
    test_prompts = [
        "Create a blue cat walking",
        "Make the jump faster",
        "Add a waving animation",
        "Export project"
    ]
    
    for prompt in test_prompts:
        result = analyzer.analyze_prompt(prompt)
        print(f"Prompt: {prompt}")
        print(f"Result: {result}")
        print()

