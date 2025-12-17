"""
Main entry point for running the Concat World Model Explorer as a module.

Usage:
    python -m concat_world_model_explorer
"""

from .app import demo

if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7861)
