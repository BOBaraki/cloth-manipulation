from setuptools import setup

setup(
    name="mujoco_utils",
    packages=["mujoco_utils"],
    version="1.0",
    install_requires=[
        "mujoco_py",
        "numpy",
    ],
    extras_require={"dev": ["black", "flake8"]},
)
