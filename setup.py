from setuptools import setup, find_packages

# Read version from src/__init__.py
with open('src/__init__.py', 'r') as f:
    for line in f:
        if line.startswith('__version__'):
            version = line.split('=')[1].strip().strip('"\'')
            break

# Read README for long description
with open('README.md', 'r') as f:
    long_description = f.read()

setup(
    name="token_geometry_analyzer",
    version=version,
    author="AI Research Team",
    author_email="research@example.com",
    description="Tool for analyzing token representations in transformer models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/token_geometry_analyzer",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.7",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.20.0",
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",
        "pandas>=1.3.0",
        "tqdm>=4.60.0",
    ],
    entry_points={
        "console_scripts": [
            "token-geometry-train=scripts.train:main",
            "token-geometry-analyze=scripts.analyze:main",
        ],
    },
)