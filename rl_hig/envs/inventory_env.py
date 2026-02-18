"""
Inventory Control (Stochastic MDP) Gymnasium Environment

State:
    s_t: inventory on-hand (Discrete: 0..S_max)

Action:
    a_t: order quantity (Discrete: 0..A_max)

Demand:
    D_t ~ Poisson(lambda) clipped to [0, D_max]  (or categorical via custom pmf)

Transition:
    pre_demand = min(S_max, s_t + a_t)
    s_{t+1} = max(0, pre_demand - D_t)

Reward (negative cost):
    cost = ordering_cost + holding_cost + shortage_cost
    reward = -cost
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any, List

import numpy as np
import gymnasium as gym
from gymnasium import spaces


@dataclass
class InventoryParams:
    # Capacities / bounds
    S_max: int = 50
    A_max: int = 50
    D_max: int = 30

    # Demand model (Poisson by default)
    demand_lambda: float = 10.0

    # Costs
    K: float = 5.0      # fixed order cost if a>0
    c: float = 1.0      # variable order cost per unit
    h: float = 0.1      # holding cost per unit of ending inventory
    p: float = 2.0      # penalty per unit of lost sales (shortage)

    # Episode
    max_steps: int = 50


class InventoryControlEnv(gym.Env):
    """
    A compact, discrete inventory control environment for RL teaching.

    Observations:
        Discrete(S_max+1) inventory on-hand

    Actions:
        Discrete(A_max+1) order quantity

    Termination:
        terminated: always False (no absorbing terminal state)
        truncated: True when step_count >= max_steps
    """

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 8}

    def __init__(
        self,
        params: InventoryParams | None = None,
        render_mode: Optional[str] = None,
        start_inventory: Optional[int] = None,
        demand_pmf: Optional[np.ndarray] = None,
    ):
        """
        Args:
            params: InventoryParams, optional.
            render_mode: "human" or "ansi" or None.
            start_inventory: if provided, reset() starts from this inventory.
            demand_pmf: optional categorical pmf over {0..D_max}. If provided,
                        overrides Poisson demand. Must sum to 1 and length D_max+1.
        """
        super().__init__()
        self.params = params or InventoryParams()
        self.render_mode = render_mode

        # Spaces
        self.observation_space = spaces.Discrete(self.params.S_max + 1)
        self.action_space = spaces.Discrete(self.params.A_max + 1)

        # Initial inventory
        if start_inventory is None:
            self._start_inventory = self.params.S_max // 2
        else:
            if not (0 <= start_inventory <= self.params.S_max):
                raise ValueError("start_inventory must be within [0, S_max].")
            self._start_inventory = int(start_inventory)

        # Demand distribution
        self._use_categorical = demand_pmf is not None
        if self._use_categorical:
            pmf = np.asarray(demand_pmf, dtype=float).copy()
            if pmf.ndim != 1 or pmf.shape[0] != self.params.D_max + 1:
                raise ValueError("demand_pmf must be a 1D array of length D_max+1.")
            s = pmf.sum()
            if not np.isfinite(s) or s <= 0:
                raise ValueError("demand_pmf must have positive finite sum.")
            pmf /= s
            if np.any(pmf < 0):
                raise ValueError("demand_pmf must be nonnegative.")
            self._demand_pmf = pmf
            self._demand_support = np.arange(self.params.D_max + 1)
        else:
            self._demand_pmf = None
            self._demand_support = None

        # Internal state
        self._s: int = 0
        self._t: int = 0
        self._last_info: Dict[str, Any] = {}

        # RNG: Gymnasium recommends using self.np_random
        self.np_random = None

    def _sample_demand(self) -> int:
        if self._use_categorical:
            d = self.np_random.choice(self._demand_support, p=self._demand_pmf)
            return int(d)
        # Poisson + clip
        d = self.np_random.poisson(lam=self.params.demand_lambda)
        if d < 0:
            d = 0
        if d > self.params.D_max:
            d = self.params.D_max
        return int(d)

    def _compute_costs(self, s: int, a: int, demand: int, s_next: int) -> Dict[str, float]:
        # fixed + variable ordering
        order_cost = (self.params.K if a > 0 else 0.0) + self.params.c * float(a)

        # holding cost based on ending inventory
        holding_cost = self.params.h * float(s_next)

        # lost sales / shortage
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

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ) -> Tuple[int, Dict[str, Any]]:
        super().reset(seed=seed)
        # Gymnasium sets self.np_random when calling super().reset(seed=...)
        self._s = int(self._start_inventory)
        self._t = 0
        self._last_info = {
            "t": self._t,
            "inventory": self._s,
        }
        if self.render_mode == "human":
            self.render()
        return self._s, dict(self._last_info)

    def step(self, action: int) -> Tuple[int, float, bool, bool, Dict[str, Any]]:
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}. Must be in [0, {self.params.A_max}].")

        a = int(action)
        s = int(self._s)

        # Demand realization
        demand = self._sample_demand()

        # Apply order, clamp to capacity
        pre_demand = s + a
        if pre_demand > self.params.S_max:
            pre_demand = self.params.S_max

        # Inventory after demand (lost sales)
        s_next = pre_demand - demand
        if s_next < 0:
            s_next = 0

        # Costs and reward
        costs = self._compute_costs(s=s, a=a, demand=demand, s_next=s_next)
        reward = -costs["total_cost"]

        # Update time/state
        self._t += 1
        self._s = int(s_next)

        terminated = False  # no absorbing terminal condition in base version
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

        if self.render_mode == "human":
            self.render()

        return self._s, float(reward), terminated, truncated, dict(info)

    def render(self) -> Optional[str]:
        """
        Render modes:
          - "ansi": returns a string
          - "human": prints a small text-based bar view (Jupyter friendly)
        """
        if self.render_mode is None:
            return None

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
                f"short={last.get('shortage_cost', 0.0):.2f} "
                f"(lost_sales={int(last.get('lost_sales', 0))})"
            )
            lines.append(f"Total cost={last.get('total_cost', 0.0):.2f} | Reward={-last.get('total_cost', 0.0):.2f}")

        out = "\n".join(lines)

        if self.render_mode == "ansi":
            return out

        # "human": print
        print(out)
        return None

    def close(self):
        return