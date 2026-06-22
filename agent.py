"""Agent: Q-network, replay buffer, action selection, and training step."""

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class QNetwork(nn.Module):
    """Maps a state to a Q-value for each action."""

    def __init__(self, obs_size, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    """Stores past experiences and samples random batches for training."""

    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)  # auto-drops oldest when full

    def push(self, state, action, reward, next_state, done):
        # save one experience tuple
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        # random batch, breaks day-to-day correlation
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(np.array(actions), dtype=torch.long),
            torch.tensor(np.array(rewards), dtype=torch.float32),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(np.array(dones), dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buffer)


def select_action(state, q_net, epsilon, n_actions):
    """Epsilon-greedy: random explore, or best known action."""
    if random.random() < epsilon:
        return random.randint(0, n_actions - 1)
    with torch.no_grad():
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        q_values = q_net(state_tensor)
        return torch.argmax(q_values).item()


def train_step(q_net, target_net, optimizer, loss_fn, buffer, batch_size=32, gamma=0.99):
    """One gradient update from a random batch of past experiences."""
    if len(buffer) < batch_size:
        return None  # not enough memories yet

    states, actions, rewards, next_states, dones = buffer.sample(batch_size)

    # Q-value the network predicted for the action actually taken
    q_values = q_net(states)
    predicted_q = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

    # best possible next-state value, from the stable target network
    with torch.no_grad():
        next_q_values = target_net(next_states)
        max_next_q = next_q_values.max(1)[0]

    # Bellman target: reward now plus discounted future value
    target_q = rewards + gamma * max_next_q * (1 - dones)

    loss = loss_fn(predicted_q, target_q)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def update_target_network(q_net, target_net):
    # copy main network's weights into the target network
    target_net.load_state_dict(q_net.state_dict())


def build_agent(obs_size, n_actions, lr=1e-3):
    """Creates q_net, target_net, optimizer, and loss function together."""
    q_net = QNetwork(obs_size, n_actions)
    target_net = QNetwork(obs_size, n_actions)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()  # only ever updated by copying, never by backprop

    optimizer = optim.Adam(q_net.parameters(), lr=lr)
    loss_fn = nn.SmoothL1Loss()  # Huber loss

    return q_net, target_net, optimizer, loss_fn
