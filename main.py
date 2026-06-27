"""Main: load data, train the agent, then step through it day by day."""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import yfinance as yf
from trading_env import TradingEnv
from agent import build_agent, select_action_and_size, train_step, update_target_network, ReplayBuffer
# map number to action
ACTION_NAMES = {0: "HOLD", 1: "BUY", 2: "SELL"}

# load stock prices and create features
def load_price_data(ticker="AAPL", period="2y"):
    # download price history and add the moving-average feature
    df = yf.Ticker(ticker).history(period=period)
    # keep only closing prices
    df = df[["Close"]].copy()
    # calculate 10 day moving average
    df["MA10"] = df["Close"].rolling(window=10).mean()
    # remove rows with missing values
    df = df.dropna()
    return df.reset_index(drop=True)

# train the DQN agent
def train(env, q_net, target_net, optimizer, loss_fn, n_actions,
          n_episodes=50, batch_size=32, target_update_freq=100):
    # epsilon greedy setting
    epsilon, epsilon_min, epsilon_decay = 1.0, 0.05, 0.97\
    # create replay buffer
    buffer = ReplayBuffer(capacity=10000)
    # track the total of steps in the environment
    step_count = 0
    loss_history = []

    # run multiple training episodes
    for episode in range(n_episodes):
        # reset environment for a new episode
        obs, info = env.reset()
        done = False
        # track total reward for this episode
        total_reward = 0

        # continue util episode done
        while not done:
            # select action based on epsilon, use epsilon-greedy exploration policy
            action, size_fraction = select_action_and_size(obs, q_net, epsilon, n_actions)
            # execute action
            next_obs, reward, terminated, truncated, info = env.step(action)
            # check whether episode is finished
            done = terminated or truncated
            # store experience in replay buffer
            buffer.push(obs, action, reward, next_obs, done)
            # train the network from replay memory and store it in loss
            loss = train_step(q_net, target_net, optimizer, loss_fn, buffer, batch_size)
            if loss is not None:               
                loss_history.append(loss)  # store in loss list
            # add step
            step_count += 1
            # update target network periodically
            if step_count % target_update_freq == 0:
                update_target_network(q_net, target_net)
            # move to next step by getting new observation
            obs = next_obs
            # accumulate reward
            total_reward += reward
        # decay once per episode
        epsilon = max(epsilon_min, epsilon * epsilon_decay)  
        print(f"Episode {episode+1}/{n_episodes} | reward={total_reward:.4f} | "
              f"epsilon={epsilon:.3f} | portfolio={info['portfolio_value']:.2f}")
        
    return loss_history

# replay the trained agent one day at a time
def run_stepper(env, q_net, n_actions):
    # play back one episode, one day at a time, fully greedy
    obs, info = env.reset()
    done = False

    # continue until episode ends
    while not done:
        # get current stock price
        close_price = env.price_data.iloc[env.current_step]["Close"]
        print(f"\nDay {env.current_step} | Price: ${close_price:.2f} | "
              f"Cash: ${env.cash:.2f} | Shares: {env.shares_held:.0f} | "
              f"Portfolio: ${info['portfolio_value']:.2f}")
        # choose best action and size based on confidence
        action, size_fraction = select_action_and_size(obs, q_net, epsilon=0.0, n_actions=n_actions)
        # execute action
        obs, reward, terminated, truncated, info = env.step(action, size_fraction)
        # check whether episode is finished
        done = terminated or truncated

       
        wanted = ACTION_NAMES[action]
        did = ACTION_NAMES[info["effective_action"]]

        # display information about what action bot takes and how much
        size_note = f" (size={size_fraction:.1%})" if did != "HOLD" else ""
        print(f"Bot wanted: {wanted} | Actually did: {did}{size_note}")
        input("Press Enter for next day...")
    # display portfolio final value
    print(f"\nFinal portfolio value: ${info['portfolio_value']:.2f}")

# start the application
if __name__ == "__main__":
    # get the price data
    df = load_price_data()
    # create the trading gymnasium environment, input the price dataframe
    env = TradingEnv(price_data=df)
    # get observation size from environment
    obs_size = env.observation_space.shape[0]
    # get number of available actions
    n_actions = env.action_space.n
    # create dqn agent
    q_net, target_net, optimizer, loss_fn = build_agent(obs_size, n_actions)
    # train the agent
    loss_history = train(env, q_net, target_net, optimizer, loss_fn, n_actions)
    # test the agent
    run_stepper(env, q_net, n_actions)

    # plot the loss
    fig, ax = plt.subplots(figsize=(8, 4.5))
    steps = np.arange(len(loss_history))
    ax.plot(steps, loss_history, color="#B8C4D9", linewidth=0.5, alpha=0.7, label="Per-update loss")
    rolling = pd.Series(loss_history).rolling(100).mean()
    ax.plot(steps, rolling, color="#2C5F8A", linewidth=2, label="100-step rolling mean")
    ax.set_yscale("log")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Huber loss (log scale)")
    ax.set_title("DQN training loss over time")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig("loss_curve.png", dpi=150)

    quarters = np.array_split(np.array(loss_history), 4)
    print("\nMean loss by training quarter:")
    for i, q in enumerate(quarters, 1):
        print(f"  Q{i}: {q.mean():.6f}")


