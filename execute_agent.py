from pettingzoo.mpe import simple_adversary_v3
import torch
from pathlib import Path

MAX_CYCLES = 500
MODEL_INDEX = 6
PRINT_INTERVAL = 10

if __name__ == "__main__":
    env = simple_adversary_v3.env(max_cycles=MAX_CYCLES, render_mode="human")
    env.metadata['render_fps'] = 200
    env.reset(seed=42)

    model_path = Path("models") / str(MODEL_INDEX)
    agent_model = torch.load(model_path / "agent_model", weights_only=False).to("cpu")
    adversary_model = torch.load(model_path / "adversary_model", weights_only=False).to("cpu")
    agent_model.eval()
    adversary_model.eval()

    iteration = 0
    cycle_num = 0
    agents_num = len(env.agents)
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()

        model = agent_model if "agent" in agent else adversary_model

        if termination or truncation:
            action = None
        else:
            action = model(torch.tensor(observation, device="cpu")).argmax().item()

        env.step(action)

        # Report rewards
        if iteration % (PRINT_INTERVAL * agents_num) == 0:
            print(f"Cycle: {cycle_num}.")
        if cycle_num % PRINT_INTERVAL == 0:
            print(f"Reward of {agent}: {reward}.")

        iteration += 1
        if iteration % agents_num == 0:
            cycle_num += 1

    env.close()
