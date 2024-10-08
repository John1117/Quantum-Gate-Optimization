# Quantum Gate Optimization - Nelder-Mead vs Bayesian Optimization

This project compares the performance of two optimization algorithms, Nelder-Mead Optimization and Bayesian Optimization, in optimizing single-qubit quantum gates, specifically Pauli-X and Hadamard gates. The goal is to determine which method is more efficient and accurate for quantum gate optimization.

## Features

- **Quantum Gate Simulation:** Pauli-X and Hadamard gate optimization.
- **Optimization Techniques:** Comparison between **Nelder-Mead Optimization** and **Bayesian Optimization**
- **Performance Metrics:** Speed, accuracy, and convergence behavior.

## Project Structure

- `optimizers/`: Contains the implementations of the Nelder-Mead and Bayesian optimizers.
- `simulator.py`: Script for simulating quantum gate behavior and running optimizations.
- `test/`: Unit tests for validating optimization results.

## Getting Started

### Prerequisites
- Python 3.x
- `numpy`
- `scipy`
- `matplotlib`
- `skopt` (for Bayesian Optimization)
