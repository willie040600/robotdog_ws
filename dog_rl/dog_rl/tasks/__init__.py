"""Gym registration of the dog locomotion tasks."""
import gymnasium as gym

gym.register(
    id="Dog-Velocity-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:DogFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.rsl_rl_ppo_cfg:DogFlatPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)

gym.register(
    id="Dog-Velocity-Flat-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:DogFlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{__name__}.rsl_rl_ppo_cfg:DogFlatPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)
