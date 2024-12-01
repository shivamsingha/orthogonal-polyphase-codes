# Orthogonal Polyphase Codes

## Overview

This repository contains a genetic algorithm implementation for generating orthogonal polyphase codes. Polyphase codes are widely used in communication systems for their ability to minimize interference and maximize signal integrity. The genetic algorithm optimizes the code sequences based on autocorrelation and cross-correlation properties, ensuring that the generated codes meet specific performance criteria.

## Features

- **Genetic Algorithm Optimization**: Utilizes a genetic algorithm to evolve a population of code sequences, optimizing for minimal sidelobe peaks and cross-correlation.
- **Parallel Processing**: Implements parallel processing to speed up fitness evaluations using Python's `multiprocessing` module.
- **Visualization**: Includes plotting functions to visualize the fitness evolution, correlation properties, and ambiguity functions of the generated codes.
- **Analysis Tools**: Provides tools to analyze the Doppler ambiguity, noise effects, and bandwidth properties of the generated codes.

## Requirements

To run this project, you will need:

- Python 3.x
- NumPy
- Matplotlib
- Seaborn
- Fractions
- Multiprocessing

You can install the required packages using pip:

```bash
pip install numpy matplotlib seaborn
```