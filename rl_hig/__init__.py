from gymnasium.envs.registration import register

register(
    id="NonlinearLQR-v0",
    entry_point="rl_hig.envs:NonlinearLQREnv",
)

register(
    id="InventoryControl-v0",
    entry_point="rl_hig.envs.inventory_env:InventoryControlEnv",
)
