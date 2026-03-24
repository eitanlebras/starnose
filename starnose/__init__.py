"""starnose — context window observability for LLM agents."""

__version__ = "0.1.0"

from starnose.sdk import trace, snapshot

__all__ = ["trace", "snapshot", "__version__"]
