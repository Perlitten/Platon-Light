from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name="platon_light",
    version="0.1.0",
    author="Platon Light Team",
    author_email="your.email@example.com",
    description="Advanced cryptocurrency trading bot with backtesting and dashboard",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Perlitten/Platon-Light",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "platon-light=run_platon_light:main",
        ],
    },
    include_package_data=True,
)
