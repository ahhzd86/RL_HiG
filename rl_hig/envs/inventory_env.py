"""
Inventory Control (Stochastic MDP) Gymnasium Environment

State:
    s_t: inventory on-hand (Discrete: 0..S_max)

Action:
    a_t: order quantity (Discrete: 0..A_max)

Demand:
    - Default: Poisson(lambda) clipped to [0, D_max]
    - Optional: categorical pmf over {0..D_max}

Transition (lost sales, no backlog):
    pre = min(S_max, s_t + a_t)
    s_{t+1} = max(0, pre - D_t)

Reward:
    reward = - (ordering_cost + holding_cost + shortage_cost)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import matplotlib.pyplot as plt

# Notebook-safe display (Jupyter/Colab). If not available, we fall back to plt.pause().
try:
    from IPython.display import display, clear_output
    _HAS_IPYTHON = True
except Exception:
    _HAS_IPYTHON = False


@dataclass
class InventoryParams:
    # Bounds / capacities
    S_max: int = 50          # max inventory capacity
    A_max: int = 50          # max order per step
    D_max: int = 30          # max demand (clipping support)

    # Demand model (Poisson default)
    demand_lambda: float = 10.0

    # Costs
    K: float = 5.0           # fixed order cost if a>0
    c: float = 1.0           # variable order cost per unit
    h: float = 0.1           # holding cost per unit of ending inventory
    p: float = 2.0           # shortage penalty per unit of lost sales

    # Episode horizon
    max_steps: int = 50


class InventoryControlEnv(gym.Env):
    """
    Discrete inventory control environment.

    observation_space: Discrete(S_max+1)
    action_space:      Discrete(A_max+1)

    terminated: always False
    truncated:  True when t >= max_steps
    """

    metadata = {"render_modes": ["human", "ansi", "plot"], "render_fps": 8}

    def __init__(
        self,
        params: Optional[InventoryParams] = None,
        render_mode: Optional[str] = None,
        start_inventory: Optional[int] = None,
        demand_pmf: Optional[np.ndarray] = None,
    ):
        super().__init__()
        self.params = params or InventoryParams()
        self.render_mode = render_mode

        # Spaces
        self.observation_space = spaces.Discrete(self.params.S_max + 1)
        self.action_space = spaces.Discrete(self.params.A_max + 1)

        # Start inventory
        if start_inventory is None:
            self._start_inventory = int(self.params.S_max // 2)
        else:
            if not (0 <= start_inventory <= self.params.S_max):
                raise ValueError("start_inventory must be in [0, S_max].")
            self._start_inventory = int(start_inventory)

        # Demand distribution
        self._use_categorical = demand_pmf is not None
        if self._use_categorical:
            pmf = np.asarray(demand_pmf, dtype=float).copy()
            if pmf.ndim != 1 or pmf.shape[0] != self.params.D_max + 1:
                raise ValueError("demand_pmf must be 1D with length D_max+1.")
            if np.any(pmf < 0):
                raise ValueError("demand_pmf must be nonnegative.")
            s = float(pmf.sum())
            if not np.isfinite(s) or s <= 0:
                raise ValueError("demand_pmf must have positive finite sum.")
            pmf /= s
            self._demand_pmf = pmf
            self._demand_support = np.arange(self.params.D_max + 1, dtype=int)
        else:
            self._demand_pmf = None
            self._demand_support = None

        # State
        self._s: int = 0
        self._t: int = 0
        self._last_info: Dict[str, Any] = {}

        # Matplotlib handles for plot mode
        self._fig = None
        self._ax = None

    # -------------------------
    # Internal helpers
    # -------------------------
    def _sample_demand(self) -> int:
        if self._use_categorical:
            d = self.np_random.choice(self._demand_support, p=self._demand_pmf)
            return int(d)

        d = self.np_random.poisson(lam=self.params.demand_lambda)
        if d < 0:
            d = 0
        if d > self.params.D_max:
            d = self.params.D_max
        return int(d)

    def _compute_costs(self, s: int, a: int, demand: int, s_next: int) -> Dict[str, float]:
        # Order cost: fixed + variable
        order_cost = (self.params.K if a > 0 else 0.0) + self.params.c * float(a)

        # Holding cost on ending inventory
        holding_cost = self.params.h * float(s_next)

        # Lost sales / shortage
        available = min(self.params.S_max, s + a)
        lost_sales = max(0, demand - available)
        shortage_cost = self.params.p * float(lost_sales)

        total_cost = order_cost + holding_cost + shortage_cost
        return {
            "order_cost": float(order_cost),
            "holding_cost": float(holding_cost),
            "shortage_cost": float(shortage_cost),
            "lost_sales": float(lost_sales),
            "total_cost": float(total_cost),
        }

    # -------------------------
    # Gym API
    # -------------------------
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[int, Dict[str, Any]]:
        super().reset(seed=seed)

        self._s = int(self._start_inventory)
        self._t = 0
        self._last_info = {"t": self._t, "inventory": self._s}

        if self.render_mode in ("human", "plot"):
            self.render()

        return self._s, dict(self._last_info)

    def step(self, action: int) -> Tuple[int, float, bool, bool, Dict[str, Any]]:
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}. Must be in [0, {self.params.A_max}].")

        a = int(action)
        s = int(self._s)

        demand = self._sample_demand()

        # Apply order, clamp to capacity
        pre = s + a
        if pre > self.params.S_max:
            pre = self.params.S_max

        # After demand (lost sales, no backlog)
        s_next = pre - demand
        if s_next < 0:
            s_next = 0

        costs = self._compute_costs(s=s, a=a, demand=demand, s_next=s_next)
        reward = -costs["total_cost"]

        self._t += 1
        self._s = int(s_next)

        terminated = False
        truncated = self._t >= self.params.max_steps

        info = {
            "t": self._t,
            "s": s,
            "a": a,
            "demand": int(demand),
            "s_next": int(s_next),
            **costs,
        }
        self._last_info = info

        if self.render_mode in ("human", "plot"):
            self.render()

        return self._s, float(reward), terminated, truncated, dict(info)

    # -------------------------
    # Rendering
    # -------------------------
    def render(self):
        if self.render_mode is None:
            return None

        # --- ANSI / human text ---
        if self.render_mode in ("human", "ansi"):
            s = self._s
            t = self._t
            last = self._last_info or {}
            demand = last.get("demand", None)
            a = last.get("a", None)

            bar_len = 40
            filled = int(round(bar_len * (s / max(1, self.params.S_max))))
            bar = "█" * filled + "·" * (bar_len - filled)

            lines: List[str] = []
            lines.append(f"InventoryControlEnv | t={t}/{self.params.max_steps}")
            lines.append(f"Inventory: {s:>3}/{self.params.S_max} | {bar}")
            if a is not None and demand is not None:
                lines.append(f"Last action (order): {a} | Last demand: {demand}")
                lines.append(
                    "Costs: "
                    f"order={last.get('order_cost', 0.0):.2f}, "
                    f"hold={last.get('holding_cost', 0.0):.2f}, "
                    f"short={last.get('shortage_cost', 0.0):.2f}, "
                    f"lost_sales={int(last.get('lost_sales', 0))}"
                )
                lines.append(
                    f"Total cost={last.get('total_cost', 0.0):.2f} | "
                    f"Reward={-last.get('total_cost', 0.0):.2f}"
                )

            out = "\n".join(lines)
            if self.render_mode == "ansi":
                return out
            print(out)
            return None

        # --- Plot mode (Jupyter/Colab-safe) ---
        if self.render_mode == "plot":
            if self._fig is None or self._ax is None:
                self._fig, self._ax = plt.subplots(figsize=(6, 4))

            self._ax.clear()

            s = self._s
            t = self._t
            last = self._last_info or {}
            demand = last.get("demand", None)
            a = last.get("a", None)

            self._ax.bar(["Inventory"], [s])
            self._ax.set_ylim(0, self.params.S_max + 10)
            self._ax.set_title(f"Inventory Level (t={t}/{self.params.max_steps})")
            self._ax.set_ylabel("Units")

            y = min(self.params.S_max + 8, s + 2)
            if a is not None:
                self._ax.text(0, y, f"Order: {a}", ha="center")
                y += 2
            if demand is not None:
                self._ax.text(0, y, f"Demand: {demand}", ha="center")

            self._fig.tight_layout()

            # Force a draw
            try:
                self._fig.canvas.draw()
            except Exception:
                pass

            if _HAS_IPYTHON:
                clear_output(wait=True)
                display(self._fig)
            else:
                plt.pause(0.01)

            return None

        return None

    def close(self):
        if self._fig is not None:
            plt.close(self._fig)
            self._fig = None
            self._ax = None
        return
