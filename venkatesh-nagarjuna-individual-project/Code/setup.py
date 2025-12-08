"""
Setup script for stock direction forecasting package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Read requirements
requirements = (this_directory / "requirements.txt").read_text().splitlines()
requirements = [r for r in requirements if not r.startswith('#') and r.strip()]

setup(
    name="stock-direction-forecasting",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Production-ready stock direction forecasting with TFT and DDG-DA",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/stock-direction-forecasting",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "flake8>=6.0.0",
            "mypy>=1.4.0",
        ],
        "aws": [
            "boto3>=1.28.0",
            "sagemaker>=2.175.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "stock-forecast-train=training.train_tft:main",
            "stock-forecast-app=app.main:main",
        ],
    },
)
