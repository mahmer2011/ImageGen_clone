"""
Command Processors

Processes structured commands from the PromptAnalyzer and executes them.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional
from flask import Flask


class CommandProcessor:
    """
    Processes commands and executes actions for the Spine animation system.
    """
    
    def __init__(self, app: Flask):
        """
        Initialize with Flask app instance.
        
        Args:
            app: Flask application instance
        """
        self.app = app
        self.jobs: Dict[str, Dict[str, Any]] = {}  # job_id -> status dict
    
    def process_command(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a structured command and return result.
        
        Args:
            command: Command dict with intent, parameters, description
            
        Returns:
            Result dict with success, message, action, and optional data
        """
        intent = command.get("intent", "unknown")
        params = command.get("parameters", {})
        description = command.get("description", "")
        
        try:
            if intent == "create_character":
                return self._handle_create_character(params, description)
            
            elif intent == "add_animation":
                return self._handle_add_animation(params, description)
            
            elif intent == "modify_animation":
                return self._handle_modify_animation(params, description)
            
            elif intent == "modify_skeleton":
                return self._handle_modify_skeleton(params, description)
            
            elif intent == "export":
                return self._handle_export(params, description)
            
            elif intent == "play_animation":
                return self._handle_play_animation(params, description)
            
            elif intent == "clarify":
                return {
                    "success": False,
                    "message": description or "I'm not sure what you want to do. Could you rephrase?",
                    "action": "clarify"
                }
            
            else:
                return {
                    "success": False,
                    "message": f"Unknown command: {intent}",
                    "action": "error"
                }
        
        except Exception as e:
            print(f"Error processing command {intent}: {e}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "message": f"Error processing command: {str(e)}",
                "action": "error"
            }
    
    def _handle_create_character(self, params: Dict[str, Any], description: str) -> Dict[str, Any]:
        """Handle character creation command."""
        char_description = params.get("description", "")
        initial_animation = params.get("initial_animation", "idle")
        
        return {
            "success": True,
            "message": f"Character creation initiated: {description or char_description}. Initial animation: {initial_animation}",
            "action": "create_character",
            "data": {
                "description": char_description,
                "initial_animation": initial_animation
            }
        }
    
    def _handle_add_animation(self, params: Dict[str, Any], description: str) -> Dict[str, Any]:
        """Handle add animation command."""
        anim_name = params.get("animation", "unknown")
        
        return {
            "success": True,
            "message": f"Adding {anim_name} animation: {description}",
            "action": "add_animation",
            "data": {
                "animation": anim_name
            }
        }
    
    def _handle_modify_animation(self, params: Dict[str, Any], description: str) -> Dict[str, Any]:
        """Handle modify animation command."""
        anim_name = params.get("animation", "unknown")
        modification = params.get("modification", "unknown")
        
        return {
            "success": True,
            "message": f"Modifying {anim_name} animation ({modification}): {description}",
            "action": "modify_animation",
            "data": params
        }
    
    def _handle_modify_skeleton(self, params: Dict[str, Any], description: str) -> Dict[str, Any]:
        """Handle modify skeleton command."""
        return {
            "success": True,
            "message": f"Skeleton modification: {description}",
            "action": "modify_skeleton",
            "data": params
        }
    
    def _handle_export(self, params: Dict[str, Any], description: str) -> Dict[str, Any]:
        """Handle export command."""
        return {
            "success": True,
            "message": "Export initiated. Your Spine project will be packaged and ready for download.",
            "action": "export",
            "data": {}
        }
    
    def _handle_play_animation(self, params: Dict[str, Any], description: str) -> Dict[str, Any]:
        """Handle play animation command."""
        anim_name = params.get("animation", "unknown")
        
        return {
            "success": True,
            "message": f"Playing {anim_name} animation: {description}",
            "action": "play_animation",
            "data": {
                "animation": anim_name
            }
        }
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """
        Get status of a background job.
        
        Args:
            job_id: Job identifier
            
        Returns:
            Status dict with status, progress, result, etc.
        """
        if job_id in self.jobs:
            return self.jobs[job_id]
        
        return {
            "status": "not_found",
            "message": f"Job {job_id} not found"
        }
    
    def create_job(self, job_id: str, initial_status: Optional[Dict[str, Any]] = None) -> None:
        """
        Create a new background job.
        
        Args:
            job_id: Job identifier
            initial_status: Optional initial status dict
        """
        self.jobs[job_id] = initial_status or {
            "status": "pending",
            "progress": 0,
            "message": "Job created"
        }
    
    def update_job(self, job_id: str, status_update: Dict[str, Any]) -> None:
        """
        Update job status.
        
        Args:
            job_id: Job identifier
            status_update: Dict with status fields to update
        """
        if job_id in self.jobs:
            self.jobs[job_id].update(status_update)
        else:
            self.jobs[job_id] = status_update


if __name__ == "__main__":
    print("Command processor module loaded successfully")

