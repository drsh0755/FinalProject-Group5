#!/usr/bin/env python
# setup.py - Package setup for FinalProject-Group5

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="finalproject-group5",
    version="0.1.0",
    author="Group 5",
    description="Deep Learning for Stock Price Prediction with Sentiment Analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mayur212626/FinalProject-Group5",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "yfinance>=0.2.0",
        "transformers>=4.30.0",
        "scikit-learn>=1.3.0",
        "nltk>=3.8.0",
        "python-dotenv>=1.0.0",
        "tqdm>=4.65.0",
        "joblib>=1.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.7.0",
            "pylint>=2.17.0",
            "mypy>=1.4.0",
        ],
        "jupyter": [
            "jupyter>=1.0.0",
            "notebook>=6.5.0",
            "ipykernel>=6.25.0",
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
        ],
    },
)
