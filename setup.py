from setuptools import setup, find_packages

setup(
    name="naviguard",
    version="0.1.0",
    description="Vision-based collision avoidance for yachts",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "ultralytics>=8.0.0",
        "numpy>=1.24.0",
        "opencv-python>=4.8.0",
        "PyYAML>=6.0",
        "scipy>=1.10.0",
        "filterpy>=1.4.5",
        "websockets>=11.0",
        "requests>=2.31.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "Pillow>=10.0.0",
        "tqdm>=4.65.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
    ],
)
