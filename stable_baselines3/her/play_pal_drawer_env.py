from stable_baselines3 import HerReplayBuffer, DDPG, SAC, TD3
from sb3_contrib import TQC
import gymnasium as gym
from robopal.demos.demo_drawer_box import DrawerEnv
from robopal.commons.gym_wrapper import GoalEnvWrapper as GymWrapper
from robopal.assets.robots.diana_med import DianaDrawer

env = DrawerEnv(
    robot=DianaDrawer(),
    renderer="viewer",
    is_render=True,
    control_freq=10,
    is_interpolate=False,
    is_pd=False,
    jnt_controller='JNTIMP',
)
env = GymWrapper(env)

# HER must be loaded with the env
model = TQC.load("./log/model_saved/TQC/diana_drawer_v1_51200.zip", env=env)

obs, info = env.reset()
for _ in range(10000):
    action, _ = model.predict(obs, deterministic=True)
    print(action)
    obs, reward, terminated, truncated, _ = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
