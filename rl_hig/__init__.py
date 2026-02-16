from gymnasium.envs.registration import register

register(
    id="NonlinearLQR-v0",
    entry_point="rl_hig.envs.nonlinear_lqr_env:NonlinearLQREnv",
)