"""Main: load data, train the agent, then step through it day by day."""

import yfinance as yf

from trading_env import TradingEnv
from agent import build_agent, select_action_and_size, train_step, update_target_network, ReplayBuffer
ACTION_NAMES = {0: "HOLD", 1: "BUY", 2: "SELL"}


def load_price_data(ticker="AAPL", period="2y"):
    # download price history and add the moving-average feature
    df = yf.Ticker(ticker).history(period=period)
    df = df[["Close"]].copy()
    df["MA10"] = df["Close"].rolling(window=10).mean()
    df = df.dropna()
    return df.reset_index(drop=True)


def train(env, q_net, target_net, optimizer, loss_fn, n_actions,
          n_episodes=50, batch_size=32, target_update_freq=100):
    # full DQN training loop, episode by episode
    epsilon, epsilon_min, epsilon_decay = 1.0, 0.05, 0.97
    buffer = ReplayBuffer(capacity=10000)
    step_count = 0

    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0

        while not done:
            action, size_fraction = select_action_and_size(obs, q_net, epsilon, n_actions)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            buffer.push(obs, action, reward, next_obs, done)
            train_step(q_net, target_net, optimizer, loss_fn, buffer, batch_size)

            step_count += 1
            if step_count % target_update_freq == 0:
                update_target_network(q_net, target_net)

            obs = next_obs
            total_reward += reward

        epsilon = max(epsilon_min, epsilon * epsilon_decay)  # decay once per episode
        print(f"Episode {episode+1}/{n_episodes} | reward={total_reward:.4f} | "
              f"epsilon={epsilon:.3f} | portfolio={info['portfolio_value']:.2f}")


def run_stepper(env, q_net, n_actions):
    # play back one episode, one day at a time, fully greedy
    obs, info = env.reset()
    done = False

    while not done:
        close_price = env.price_data.iloc[env.current_step]["Close"]
        print(f"\nDay {env.current_step} | Price: ${close_price:.2f} | "
              f"Cash: ${env.cash:.2f} | Shares: {env.shares_held:.0f} | "
              f"Portfolio: ${info['portfolio_value']:.2f}")

        action, size_fraction = select_action_and_size(obs, q_net, epsilon=0.0, n_actions=n_actions)
        obs, reward, terminated, truncated, info = env.step(action, size_fraction)
        done = terminated or truncated

        wanted = ACTION_NAMES[action]
        did = ACTION_NAMES[info["effective_action"]]
        size_note = f" (size={size_fraction:.1%})" if did != "HOLD" else ""
        print(f"Bot wanted: {wanted} | Actually did: {did}{size_note}")

        input("Press Enter for next day...")

    print(f"\nFinal portfolio value: ${info['portfolio_value']:.2f}")


if __name__ == "__main__":
    df = load_price_data()
    env = TradingEnv(price_data=df)

    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n
    q_net, target_net, optimizer, loss_fn = build_agent(obs_size, n_actions)

    train(env, q_net, target_net, optimizer, loss_fn, n_actions)
    run_stepper(env, q_net, n_actions)
