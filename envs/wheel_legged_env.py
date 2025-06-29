import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from gym import Env, spaces

from controllers.pd_controller import PDController


@dataclass
class WheelLeggedEnv(Env):
    """Simple wheel-legged robot environment using PD control."""

    dt: float = 0.02
    g: float = 9.81
    m_body: float = 1.0
    m_wheel: float = 0.5
    l: float = 0.5
    max_torque: float = 2.0
    render_mode: Optional[str] = None

    def __post_init__(self):
        high = np.array([np.pi, 10.0, np.pi, 10.0], dtype=np.float32)
        self.action_space = spaces.Box(
            low=-math.pi, high=math.pi, shape=(1,), dtype=np.float32
        )
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.state = None
        self.controller = PDController(kp=20.0, kd=5.0)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        return np.array(self.state, dtype=np.float32), {}

    def step(self, action: np.ndarray):
        target_angle = float(np.clip(action[0], -math.pi, math.pi))
        x, x_dot, theta, theta_dot = self.state
        error = target_angle - theta
        error_rate = -theta_dot
        torque = np.clip(
            self.controller.compute_torque(error, error_rate),
            -self.max_torque,
            self.max_torque,
        )
        sin_theta = math.sin(theta)
        cos_theta = math.cos(theta)
        total_mass = self.m_body + self.m_wheel
        temp = (torque + self.m_body * self.l * theta_dot ** 2 * sin_theta) / total_mass
        thetaacc = (
            self.g * sin_theta - cos_theta * temp
        ) / (self.l * (4.0 / 3.0 - self.m_body * cos_theta ** 2 / total_mass))
        xacc = temp - self.m_body * self.l * thetaacc * cos_theta / total_mass
        x += self.dt * x_dot
        x_dot += self.dt * xacc
        theta += self.dt * theta_dot
        theta_dot += self.dt * thetaacc
        self.state = (x, x_dot, theta, theta_dot)
        reward = 1.0 - (theta ** 2 + 0.1 * theta_dot ** 2)
        terminated = bool(theta < -math.pi / 2 or theta > math.pi / 2)
        return np.array(self.state, dtype=np.float32), reward, terminated, False, {}

    def render(self):
        if self.render_mode == "human":
            print(f"state: {self.state}")

    def close(self):
        pass
