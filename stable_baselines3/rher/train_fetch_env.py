import gymnasium as gym
import numpy as np

from stable_baselines3 import HerReplayBuffer, DDPG
from stable_baselines3 import RHerReplayBuffer, DDPG_RHER
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import NormalActionNoise


class TensorboardCallback(BaseCallback):
    def __init__(self, log_dir, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        self.log_dir = log_dir

    def _on_step(self) -> bool:
        if self.n_calls % 10000 == 0:
            self.model.save(self.log_dir + f"/model_saved/DDPG/fetch_push_{self.n_calls}")
        return True


log_dir = "log/"

env = gym.make('FetchPush-v2', render_mode=None)

# The noise objects for DDPG
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
# Initialize the model
model = DDPG_RHER(
    'MultiInputPolicy',
    env,
    replay_buffer_class=RHerReplayBuffer,
    # Parameters for HER
    replay_buffer_kwargs=dict(
        n_sampled_goal=4,
        goal_selection_strategy="future",
    ),
    verbose=1,
    tensorboard_log=log_dir,
    action_noise=action_noise,
    batch_size=256,
    gamma=0.98,
    tau=0.05,
    learning_starts=100,
    policy_kwargs=dict(n_critics=1, net_arch=[256]),
)

# Train the model
model.learn(int(1e6), callback=TensorboardCallback(log_dir=log_dir))

model.save("./her_bit_env")
