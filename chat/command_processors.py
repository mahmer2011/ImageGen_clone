"""
Command Processors

Executes structured commands on the Spine animation system.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


class CommandProcessor:
    """
    Processes structured commands and executes them.
    """
    
    def __init__(self, app_context: Any):
        """
        Initialize with Flask app context or similar.
        
        Args:
            app_context: Application context for accessing resources
        """
        self.app_context = app_context
        self.current_character_id = None
        self.current_spine_dir = None
    
    def process_command(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a parsed command and execute the appropriate action.
        
        Args:
            command: Parsed command dict from PromptAnalyzer
            
        Returns:
            Result dict with success status and message
        """
        intent = command.get("intent")
        parameters = command.get("parameters", {})
        
        try:
            if intent == "create_character":
                return self._create_character(parameters)
            
            elif intent == "add_animation":
                return self._add_animation(parameters)
            
            elif intent == "modify_animation":
                return self._modify_animation(parameters)
            
            elif intent == "modify_skeleton":
                return self._modify_skeleton(parameters)
            
            elif intent == "export":
                return self._export_project(parameters)
            
            elif intent == "play_animation":
                return self._play_animation(parameters)
            
            elif intent == "clarify":
                return {
                    "success": False,
                    "message": "I didn't understand that command. Try something like: 'create a blue cat', 'make the jump faster', or 'add a wave animation'",
                    "action": "clarify"
                }
            
            else:
                return {
                    "success": False,
                    "message": f"Unknown intent: {intent}",
                    "action": "error"
                }
        
        except Exception as e:
            return {
                "success": False,
                "message": f"Error executing command: {str(e)}",
                "action": "error"
            }
    
    def _create_character(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle character creation command.
        """
        description = params.get("description", "character")
        
        return {
            "success": True,
            "message": f"To create '{description}', please use the main interface to:\n1. Enter prompt\n2. Enhance prompt\n3. Generate image\n4. Segment image\n5. Create Spine skeleton",
            "action": "guide",
            "steps": ["enhance", "generate", "segment", "create_spine"]
        }
    
    def _add_animation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle add animation command.
        """
        animation = params.get("animation")
        
        if not animation:
            return {"success": False, "message": "Animation name not specified", "action": "error"}
        
        # Animations are already included in default creation
        return {
            "success": True,
            "message": f"The '{animation}' animation is already included! Click the '{animation.title()}' button to play it.",
            "action": "play",
            "animation": animation
        }
    
    def _modify_animation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle animation modification command.
        """
        animation = params.get("animation")
        modification = params.get("modification")
        
        if modification == "speed":
            factor = params.get("factor", 1.5)
            return {
                "success": True,
                "message": f"Animation speed adjustment would be applied (factor: {factor}x). This requires re-generating the Spine skeleton.",
                "action": "info"
            }
        
        return {
            "success": True,
            "message": "Animation modifications are applied when creating the Spine skeleton.",
            "action": "info"
        }
    
    def _modify_skeleton(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle skeleton modification command.
        """
        description = params.get("description", "skeleton")
        
        return {
            "success": True,
            "message": "Skeleton modifications require re-generating the character. Please generate a new image with the desired proportions.",
            "action": "info"
        }
    
    def _export_project(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle export command.
        """
        return {
            "success": True,
            "message": "Click the 'Export Spine Project' button below the animation viewer to download your project.",
            "action": "guide"
        }
    
    def _play_animation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle play animation command.
        """
        animation = params.get("animation")
        
        return {
            "success": True,
            "message": f"Click the '{animation.title()}' button to play the animation.",
            "action": "play",
            "animation": animation
        }


if __name__ == "__main__":
    print("Command processor module loaded successfully")

