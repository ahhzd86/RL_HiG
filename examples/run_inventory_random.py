import time
import gymnasium as gym
import rl_hig  # noqa: F401 (needed so envs get registered)

def main():
    env = gym.make("InventoryControl-v0", render_mode="human")
    obs, info = env.reset(seed=0)

    total_reward = 0.0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        time.sleep(0.1)

    print("\nEpisode finished.")
    print("Total reward:", total_reward)
    env.close()

if __name__ == "__main__":
    main()