from __future__ import annotations
import matplotlib.pyplot as plt
from IPython.display import clear_output, display

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class NonlinearLQREnv(gym.Env):
    """
    Nonlinear 1D control system with LQR-like quadratic cost.

    Dynamics:
        x_{t+1} = a*x_t + b*u_t + c*x_t^3 + w_t
        w_t ~ N(0, sigma^2)

    Reward:
        r_{t+1} = - (q*x_t^2 + r*u_t^2)
    """

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self,
        *,
        a: float = 1.0,
        b: float = 0.5,
        c: float = 0.15,
        q: float = 1.0,
        r: float = 0.05,
        gamma: float = 0.99,
        sigma: float = 0.05,
        x_clip: float = 4.0,
        u_clip: float = 3.0,
        max_steps: int = 200,
        init_low: float = -1.0,
        init_high: float = 1.0,
        termination_threshold: float | None = 4.0,
        render_mode: str | None = None,
    ):
        super().__init__()

        self.a = float(a)
        self.b = float(b)
        self.c = float(c)
        self.q = float(q)
        self.r = float(r)
        self.gamma = float(gamma)
        self.sigma = float(sigma)

        self.x_clip = float(x_clip)
        self.u_clip = float(u_clip)
        self.max_steps = int(max_steps)

        self.init_low = float(init_low)
        self.init_high = float(init_high)

        self.termination_threshold = termination_threshold
        self.render_mode = render_mode

        self.observation_space = spaces.Box(
            low=np.array([-self.x_clip], dtype=np.float32),
            high=np.array([self.x_clip], dtype=np.float32),
            dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=np.array([-self.u_clip], dtype=np.float32),
            high=np.array([self.u_clip], dtype=np.float32),
            dtype=np.float32,
        )

        self._x: float = 0.0
        self._t: int = 0

        self._x_hist = []
        self._u_hist = []
        
        self._fig = None
        self._ax = None

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._t = 0
        self._x = float(self.np_random.uniform(self.init_low, self.init_high))
        obs = np.array([self._x], dtype=np.float32)
        self._x_hist = [self._x]
        self._u_hist = [0.0]
        info = {"t": self._t, "x": self._x}
        return obs, info

    def step(self, action):
        self._t += 1

        u = float(np.asarray(action).reshape(-1)[0])
        u = float(np.clip(u, -self.u_clip, self.u_clip))

        reward = - (self.q * (self._x ** 2) + self.r * (u ** 2))

        w = float(self.np_random.normal(loc=0.0, scale=self.sigma))
        x_next = self.a * self._x + self.b * u + self.c * (self._x ** 3) + w

        self._x = float(np.clip(x_next, -self.x_clip, self.x_clip))
        self._x_hist.append(self._x)
        self._u_hist.append(u)

        terminated = False
        if self.termination_threshold is not None and abs(x_next) > float(self.termination_threshold):
            terminated = True

        truncated = self._t >= self.max_steps

        obs = np.array([self._x], dtype=np.float32)
        info = {"t": self._t, "x": self._x, "u": u, "w": w, "raw_x_next": x_next, "cost": -reward}

        if self.render_mode == "human":
            self.render()

        return obs, float(reward), terminated, truncated, info

    def render(self):
        if self._fig is None:
            self._fig, self._ax = plt.subplots(2, 1, figsize=(7, 4), sharex=True)
    
        t = np.arange(len(self._x_hist))
    
        self._ax[0].cla()
        self._ax[0].plot(t, self._x_hist)
        self._ax[0].set_ylabel("state x")
        self._ax[0].grid(True)
    
        self._ax[1].cla()
        self._ax[1].plot(t, self._u_hist)
        self._ax[1].set_ylabel("action u")
        self._ax[1].set_xlabel("time step")
        self._ax[1].grid(True)
    
        self._fig.suptitle(f"NonlinearLQREnv | t={self._t}")
        self._fig.tight_layout()
    
        clear_output(wait=True)
        display(self._fig)
    
        # Prevent duplicate display in Colab
        plt.close(self._fig)
    
        def close(self):

        pass



