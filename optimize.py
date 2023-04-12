from custom_env import Reflex_WALK_Env
from reflex_opt.optimize import optimize
import os

if __name__ == '__main__':
    path=os.getcwd()+'/assets'
    VPenv = Reflex_WALK_Env(path=path)
    optimize(env=VPenv)
