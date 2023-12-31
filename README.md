# Differential Evolution

This repository contains code for evaluating optimization algorithms on sample problems. It includes an implementation of differential evolution

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes:

### Prerequisites

You will need Python 3 and pip installed. 

### Installing dependencies

To install the requirements, run:

```bash
pip install -r requirements.txt
```

This will install any dependencies needed to run the code.

### Directory structure

- src/
  - optimizers.py # contains DifferentialEvolution class
  - utilis.py # contains utility functions
- example.ipynb # Jupyter notebook demonstrating usage
- requirements.txt # list of dependencies

## Usage

The optimizers.py module contains an implementation of the DifferentialEvolution optimizer. Utility functions used by the optimizers are in utilis.py.

An example Jupyter notebook (example.ipynb) shows how to use the DifferentialEvolution class to optimize a sample function.
