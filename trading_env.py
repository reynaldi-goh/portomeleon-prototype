"""Trading environment: holds price data, portfolio state, and step logic."""

import gymnasium as gym
import numpy as np


class TradingEnv(gym.Env):
    def __init__(self, price_data, initial_cash=10000):
        super().__init__()
        self.price_data = price_data
        self.initial_cash = initial_cash

        # 3 actions: hold, buy, sell
        self.action_space = gym.spaces.Discrete(3)
        # 5 numbers: Close, MA10, cash, shares_held, portfolio_value
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32
        )

    def _get_obs(self):
        # build the state vector for the current day, normalized to comparable scales
        row = self.price_data.iloc[self.current_step]
        price = row["Close"]
        return np.array([
            price / self.initial_price,
            row["MA10"] / self.initial_price,
            self.cash / self.initial_cash,
            (self.shares_held * price) / self.initial_cash,
            self.portfolio_value / self.initial_cash,
        ], dtype=np.float32)

    def _get_info(self):
        # extra info: value plus what actually happened
        return {
            "portfolio_value": self.portfolio_value,
            "effective_action": self.effective_action,
        }

    def reset(self, seed=None, options=None):
        # start a fresh episode from day 0
        super().reset(seed=seed)
        self.current_step = 0
        self.cash = self.initial_cash
        self.shares_held = 0
        self.portfolio_value = self.initial_cash
        self.effective_action = 0
        self.initial_price = self.price_data.iloc[0]["Close"]
        return self._get_obs(), self._get_info()

    def step(self, action, size_fraction=1.0):
        price = self.price_data.iloc[self.current_step]["Close"]

        # apply action, but only if it actually changes anything
        if action == 1:  # buy
            cash_to_spend = self.cash * size_fraction
            shares_to_buy = cash_to_spend // price
            if shares_to_buy > 0:
                self.cash -= shares_to_buy * price
                self.shares_held += shares_to_buy
                self.effective_action = 1
            else:
                self.effective_action = 0  # no cash, counts as hold
        elif action == 2:  # sell
            shares_to_sell = self.shares_held * size_fraction
            if shares_to_sell > 0:
                self.cash += shares_to_sell * price
                self.shares_held -= shares_to_sell
                self.effective_action = 2
            else:
                self.effective_action = 0  # nothing to sell, counts as hold
        else:
            self.effective_action = 0  # explicit hold

        prev_value = self.portfolio_value
        self.portfolio_value = self.cash + self.shares_held * price

        self.current_step += 1  # advance to next day

        # reward: % change in portfolio value this step
        reward = (self.portfolio_value - prev_value) / prev_value

        terminated = False
        truncated = self.current_step >= len(self.price_data) - 1

        return self._get_obs(), reward, terminated, truncated, self._get_info()
