"""Sport-specific edge consumers built on engine score distributions."""

# Import the consumer implementations so they self-register in EDGE_CONSUMERS at
# package-import time (mirrors how engine.py registers GAME_BACKENDS). Any import
# of an ``omega.core.edge`` submodule runs this and populates the registry.
from omega.core.edge import soccer_consumer as soccer_consumer  # noqa: F401
