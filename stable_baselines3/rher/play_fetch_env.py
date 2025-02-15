from stable_baselines3 import HerReplayBuffer, DDPG
from stable_baselines3 import RHerReplayBuffer, DDPG_RHER
import gymnasium as gym

# env = gym.make('FetchPickAndPlace-v2', render_mode='human')
env = gym.make('FetchPush-v2', render_mode='human')

model = DDPG_RHER.load("./log/model_saved/DDPG/fetch_push_30000.zip", env=env)

obs, info = env.reset()
obs['desired_goal'] *= 0
for _ in range(100000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    obs['desired_goal'] *= 0
    if terminated or truncated:
        obs, info = env.reset()
        obs['desired_goal'] *= 0
