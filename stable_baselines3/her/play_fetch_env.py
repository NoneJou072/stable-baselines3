from stable_baselines3 import HerReplayBuffer, DDPG, SAC, TD3
from sb3_contrib import TQC
import gymnasium as gym

env = gym.make('FetchPickAndPlace-v2', render_mode='human')

# HER must be loaded with the env
model = TQC.load("./log/model_saved/TQC/fetch_pick_place_204800.zip", env=env)

obs, info = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    print(info)
    if terminated or truncated:
        obs, info = env.reset()
