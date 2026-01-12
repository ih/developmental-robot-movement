"""
Main entry point for running the Concat World Model Explorer as a module.

Usage:
    python -m concat_world_model_explorer              # Auto-find available port
    python -m concat_world_model_explorer --port 7862  # Use specific port
"""

import argparse
import socket


def find_available_port(start_port=7861, max_attempts=100):
    """Find an available port starting from start_port."""
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"Could not find available port in range {start_port}-{start_port + max_attempts}")


def main():
    parser = argparse.ArgumentParser(
        description="Concat World Model Explorer - Web UI for training world models"
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=None,
        help="Port to run the server on. If not specified, finds an available port starting from 7861."
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public Gradio link"
    )
    args = parser.parse_args()

    # Determine port
    if args.port is not None:
        port = args.port
    else:
        port = find_available_port()

    # Import here to avoid slow import if just checking --help
    from . import state
    from .app import demo

    # Set instance ID based on port for unique checkpoint naming
    state.instance_id = f"p{port}"
    print(f"Instance ID: {state.instance_id}")

    print(f"Starting Concat World Model Explorer on port {port}...")
    demo.launch(share=args.share, server_name="0.0.0.0", server_port=port)


if __name__ == "__main__":
    main()
