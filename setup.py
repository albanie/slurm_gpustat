import setuptools


with open("README.md", "r") as f:
    long_description = f.read()


setuptools.setup(
    name="slurm_gpustat",
    version="0.0.3.9",
    entry_points={
        "console_scripts": [
            "slurm_gpustat=slurm_gpustat.slurm_gpustat:main",
        ],
    },
    author="Samuel Albanie",
    description="A simple SLURM gpu summary tool",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/albanie/slurm_gpustat",
    packages=["slurm_gpustat"],
    python_requires=">=3.6",
    install_requires=[
        "numpy",
        "colored",
        "seaborn",
        "zsvision",
        "scipy>=1.4.1"
    ],
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
        'Operating System :: POSIX :: Linux',
    ],
)
