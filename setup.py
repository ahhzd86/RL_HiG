from setuptools import setup, find_packages

setup(
    name="rl_hig",
    version="0.0.5",
    description="Teaching library for RL (bandits + custom Gymnasium envs).",
    packages=find_packages(),
    py_modules=["hidden_bandit"],
    install_requires=[
        "numpy>=1.26,<2.1",
        "gymnasium>=0.29",
    ],
    python_requires=">=3.9",
)