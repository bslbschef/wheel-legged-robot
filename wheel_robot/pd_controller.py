class PDController:
    """Simple Proportional-Derivative (PD) controller."""

    def __init__(self, kp: float, kd: float):
        self.kp = kp
        self.kd = kd

    def compute(self, target: float, current: float, target_velocity: float = 0.0, current_velocity: float = 0.0) -> float:
        """Compute the control output."""
        error = target - current
        error_dot = target_velocity - current_velocity
        return self.kp * error + self.kd * error_dot
