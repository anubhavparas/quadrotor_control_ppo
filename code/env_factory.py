from constants import HOVER_ENV, TRAJ_ENV
from hover_task import HoverTaskEnv
from traj_task import TrajTaskEnv

env_map = {
    HOVER_ENV: HoverTaskEnv(),
    TRAJ_ENV: TrajTaskEnv()
}


def make(env):
    return env_map[env]
