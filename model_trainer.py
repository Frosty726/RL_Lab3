from lib.DQN import DQN
from lib.HistoryBuffer import HistoryBuffer, Transition

import torch
from torch import nn
from torch.optim import Adam

from pettingzoo.mpe import simple_adversary_v3
import random
from pathlib import Path
import json
import math


RANDOM_SEED = 42

BATCH_SIZE = 32
LEARNING_RATE = 0.01
MAX_CYCLES = 5000

EPSILON_MAX = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 10000

GAMMA = 0.99
TAU = 0.005

HISTORY_BUFFER_SIZE = 100

def train():
    # Define target device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} to train models.")

    # Initialize environment
    env = simple_adversary_v3.env(max_cycles=MAX_CYCLES)
    env.reset(seed=RANDOM_SEED)

    # Read env variables
    good_obs_num = 0
    good_acts_num = 0
    adversary_obs_num = 0
    adversary_acts_num = 0
    for agent in env.agents:
        if "agent" in agent:
            good_obs_num = len(env.observe(agent))
            good_acts_num = env.action_space(agent).n
        if "adversary" in agent:
            adversary_obs_num = len(env.observe(agent))
            adversary_acts_num = env.action_space(agent).n

    # Agent transitions history
    # Batches are sampled from here later
    history_buffer = {agent : HistoryBuffer(HISTORY_BUFFER_SIZE) for agent in env.agents}

    # Initialize models
    agent_model = DQN(in_feats=good_obs_num, out_feats=good_acts_num).to(device)
    adversary_model = DQN(in_feats=adversary_obs_num, out_feats=adversary_acts_num).to(device)

    agent_model_target = DQN(in_feats=good_obs_num, out_feats=good_acts_num).to(device)
    adversary_model_target = DQN(in_feats=adversary_obs_num, out_feats=adversary_acts_num).to(device)
    agent_model_target.load_state_dict(agent_model.state_dict())
    adversary_model_target.load_state_dict(adversary_model.state_dict())

    # Initialize optimizers
    agent_optimizer = Adam(agent_model.parameters(), lr=LEARNING_RATE)
    adversary_optimizer = Adam(adversary_model.parameters(), lr=LEARNING_RATE)

    # Main cycle
    iteration = 0
    cycle_num = 0
    agents_number = len(env.agents)
    for agent in env.agent_iter():
        if iteration % (100 * agents_number) == 0:
            print(f"Iteration {cycle_num}\t of {MAX_CYCLES}.")

        policy_model = agent_model if "agent" in agent else adversary_model
        target_model = agent_model_target if "agent" in agent else adversary_model_target
        optimizer = agent_optimizer if "agent" in agent else adversary_optimizer

        # Read old state
        state, reward, terminated, truncated, _ = env.last()
        done = terminated or truncated
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        if done:
            env.step(None)
        else:
            epsilon_threshold = EPSILON_MIN + (EPSILON_MAX - EPSILON_MIN) * math.exp(-1. * cycle_num / EPSILON_DECAY)
            if random.random() > epsilon_threshold:
                action = select_action(policy_model, state)
            else:
                action = env.action_space(agent).sample()
                action = torch.tensor([[action]], device=device, dtype=torch.long)
            env.step(action.item())

            observation = env.observe(agent)
            reward = env.rewards[agent] - reward
            reward = torch.tensor([reward], device=device)

            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            # Store the transition in memory
            history_buffer[agent].push(state, action, next_state, reward)

            # Perform one step of the optimization (on the policy network)
            optimize_model(policy_model, target_model, optimizer, history_buffer[agent], device)

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_model.state_dict()
            policy_net_state_dict = policy_model.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
            target_model.load_state_dict(target_net_state_dict)

        iteration += 1
        if iteration % agents_number == 0:
            cycle_num += 1

    return agent_model, adversary_model

def select_action(policy_model, state):
    with torch.no_grad():
        # t.max(1) will return the largest column value of each row.
        # second column on max result is index of where max element was
        # found, so we pick action with the larger expected reward.
        action = policy_model(state).max(1).indices.view(1, 1)

    return action

def optimize_model(policy_model, target_model, optimizer, history : HistoryBuffer, device):
    if len(history) < BATCH_SIZE:
        return

    transitions = history.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_model(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_model(non_final_next_states).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_model.parameters(), 100)
    optimizer.step()

def save_models(agent_model : nn.Module, adversary_model : nn.Module):
    models_path = Path("models")
    models_path.mkdir(exist_ok=True)
    model_ids = [0]
    for model_dir in models_path.iterdir():
        model_ids.append(int(model_dir.name))
    current_index = max(model_ids) + 1
    model_path = models_path / str(current_index)
    model_path.mkdir()
    torch.save(agent_model, model_path / "agent_model")
    torch.save(adversary_model, model_path / "adversary_model")
    parameters = {
        "random_seed" : RANDOM_SEED,

        "batch_size" : BATCH_SIZE,
        "learning_rate" : LEARNING_RATE,
        "max_cycles" : MAX_CYCLES,

        "epsilon_max" : EPSILON_MAX,
        "epsilon_min" : EPSILON_MIN,
        "epsilon_decay" : EPSILON_DECAY,

        "gamma" : GAMMA,
        "tau" : TAU,

        "history_buffer_size" : HISTORY_BUFFER_SIZE,
    }
    with open(model_path / "parameters.json", "w") as f:
        json.dump(parameters, f)

if __name__ == "__main__":
    random.seed(RANDOM_SEED)
    agent_model, adversary_model = train()
    save_models(agent_model, adversary_model)