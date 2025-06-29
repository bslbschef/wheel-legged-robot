class WheelLeggedEnv:
    """Minimal wheel-legged robot environment placeholder."""

    def __init__(self, config=None):
        self.config = config or {}
        self.initialized = True

    def reset(self):
        """Reset the environment state."""
        self.state = 0
        return self.state
