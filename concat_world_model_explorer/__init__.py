"""
Concat World Model Explorer - A web-based UI for exploring world models.

This package provides a Gradio-based interface for running AutoencoderConcatPredictorWorldModel
on recorded robot sessions with comprehensive visualization and analysis tools.
"""

# Import the demo object when app.py is created
# This allows running the app with: python -m concat_world_model_explorer
try:
    from .app import demo
    __all__ = ['demo']
except ImportError:
    # app.py doesn't exist yet
    demo = None
    __all__ = []
