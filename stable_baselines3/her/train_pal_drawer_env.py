from stable_baselines3 import HerReplayBuffer, DDPG
from sb3_contrib import TQC
from stable_baselines3.common.callbacks import BaseCallback
from robopal.demos.demo_drawer_box import DrawerEnv
from robopal.commons.gym_wrapper import GoalEnvWrapper as GymWrapper


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, log_dir, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        self.log_dir = log_dir

    def _on_step(self) -> bool:
        if self.n_calls % 10000 == 0:
            self.model.save(self.log_dir + f"/model_saved/TQC/diana_drawer_v1_{self.n_calls}")
        return True


log_dir = "log/"

from robopal.assets.robots.diana_med import DianaDrawer

env = DrawerEnv(
    robot=DianaDrawer(),
    renderer="viewer",
    is_render=False,
    control_freq=10,
    is_interpolate=False,
    is_pd=False,
    jnt_controller='JNTIMP',
)
env = GymWrapper(env)

# Initialize the model
model = TQC(
    'MultiInputPolicy',
    env,
    replay_buffer_class=HerReplayBuffer,
    # Parameters for HER
    replay_buffer_kwargs=dict(
        n_sampled_goal=4,
        goal_selection_strategy="future",
    ),
    verbose=1,
    tensorboard_log=log_dir,
    batch_size=1024,
    gamma=0.95,
    tau=0.05,
    policy_kwargs=dict(n_critics=2, net_arch=[512, 512, 512]),
)

# Train the model
model.learn(int(1e6), callback=TensorboardCallback(log_dir=log_dir))

model.save("./her_bit_env")
