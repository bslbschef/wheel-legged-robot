"""PD controller module for controlling wheel-legged robot.

This module provides a simple proportional-derivative (PD) controller
implementation. Instantiate :class:`PDController` with proportional
and derivative gains, then call :meth:`compute_torque` with the current
error and error derivative to obtain a control torque.
"""

from dataclasses import dataclass

@dataclass
class PDController:
    """Simple PD controller.

    Parameters
    ----------
    kp : float
        Proportional gain.
    kd : float
        Derivative gain.
    """

    kp: float
    kd: float

    def compute_torque(self, error: float, error_rate: float) -> float:
        """Compute the control torque.

        Parameters
        ----------
        error : float
            Position or angle error.
        error_rate : float
            Rate of change of the error (derivative).

        Returns
        -------
        float
            The computed control torque.
        """
        return self.kp * error + self.kd * error_rate
