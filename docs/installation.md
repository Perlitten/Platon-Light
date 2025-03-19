# Installation Guide

This guide will walk you through the process of installing Platon Light on your system.

## Prerequisites

Before installing Platon Light, make sure you have the following prerequisites:

- Python 3.9 or higher
- pip (Python package installer)
- Git (optional, for cloning the repository)

## Installation Steps

### 1. Clone the Repository (or Download)

```bash
git clone https://github.com/Perlitten/Platon-Light.git
cd Platon-Light
```

Alternatively, you can download the repository as a ZIP file from GitHub and extract it.

### 2. Install Dependencies

Install all required dependencies using pip:

```bash
pip install -r requirements.txt
```

### 3. Set Up Environment Variables

Create a `.env` file in the root directory of the project based on the provided `.env.example` file:

```bash
# Copy the example file
cp .env.example .env

# Edit the .env file with your preferred text editor
# Replace the placeholder values with your actual API keys and settings
```

### 4. Verify Installation

Run the following command to verify that Platon Light has been installed correctly:

```bash
python run_platon_light.py
```

This should display the main menu of Platon Light.

## Installing as a Package (Optional)

If you want to install Platon Light as a Python package, you can use the following command:

```bash
pip install -e .
```

This will install Platon Light in development mode, allowing you to make changes to the code and have them immediately reflected without reinstalling.

## Troubleshooting

If you encounter any issues during installation, please check the following:

- Make sure you have the correct Python version installed
- Check that all dependencies were installed correctly
- Verify that your `.env` file is properly configured

If you continue to experience issues, please [open an issue](https://github.com/Perlitten/Platon-Light/issues) on GitHub.
