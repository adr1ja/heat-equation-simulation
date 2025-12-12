# 1D Heat Equation Solver

This project implements a numerical solution for the one-dimensional heat equation using the Finite Difference Method (Forward-Time Central-Space scheme).

It allows users to define simulation parameters (like diffusivity, length, and time steps) via a configuration file and outputs the temperature distribution over time.

## ⚠️ AI Generation Disclaimer
**I acknowledge the use of RAI (https://rai.uni-stuttgart.de/) to implement the entirety of this code.**
* **Tool:** HAWKI (OpenAI GPT-OSS 120B)
* **Prompt:** "Write a Python script that solves the 1D heat equation. The script should read parameters (like diffusivity, time steps, length) from a configuration file (e.g., JSON or YAML) and save the results to an output file. Please include a basic test function and docstrings explaining the code."

## Features
* Solves the 1D Heat Equation: $\frac{\partial u}{\partial t} = \alpha \frac{\partial^2 u}{\partial x^2}$
* Configurable via JSON or YAML.
* Exports results to CSV or NumPy binary files.
* Includes basic unit tests.

## Requirements
* Python 3.x
* NumPy
* PyYAML (optional, for YAML config support)

You can install dependencies using:
```bash
pip install numpy pyyaml
```
## Usage
1. Create a configuration file (e.g., config.json):
```bash
{
    "diffusivity": 1.0,
    "length": 1.0,
    "nx": 50,
    "dt": 0.0001,
    "t_end": 0.05
}
```
2. Run the simulation:
```bash
python heat1d.py config.json
```
3. Check results: The output will be saved to results.csv by default.

##Testing
To run the included tests, use pytest:
```bash
pytest test_heat1d.py
```
