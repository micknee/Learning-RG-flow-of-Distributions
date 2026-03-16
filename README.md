# Learning-RG-flow-of-Distributions
A JAX implementation of normalizing flows for learning the Renormalization Group (RG) flow of probability distributions over Standard Model coupling constants

## Overview

Renormalization Group (RG) determines how coupling constants in a Quantum Field Theory evolve between energy scales. Rather than evolving individual trajectories, this project treats the couplings as random variables and learns the full probability distribution over IR couplings conditioned on a UV distribution, extending the analytical framework of [Eichhorn & Held (2025)](https://arxiv.org/abs/2506.12548).

The notebook demonstrates:
1. **1-parameter toy model**: Monte Carlo sampling of a UV distribution and numerical RG flow of the U(1) gauge coupling α_Y, compared to the analytical solution
2. **7-parameter Standard Model**: RG flow of all third-generation SM couplings (α_1, α_2, α_3, λ, y_t, y_b, y_τ) at two-loop order from the Planck scale to the electroweak scale
3. **Conditional normalizing flow**: A trained flow that maps any UV distribution (specified by a mean and covariance matrix) to the corresponding IR distribution, enabling fast amortized inference without re-running the ODE solver

## Key Results

- The trained flow reproduces the marginal distributions and correlation structure of the exact Monte Carlo RG evolution and generalises to unseen UV distributions
- Sampling from the flow is ~3.9x faster than running the ODE integration, with the speedup growing as the physical model becomes more complex

## Installation

Clone the repository and install dependencies:

git clone https://github.com/micknee/rg-flow-distributions
cd rg-flow-distributions
pip install -r requirements.txt

## Usage

Open and run RG_Flow_of_Distributions.ipynb sequentially. The notebook is structured as:

| Section | Description |
|---------|-------------|
| Imports | Library imports and JAX configuration |
| Toy model | 1-parameter U(1) example with analytical comparison |
| Full SM model | Beta functions, ODE solver for RG evolution, sampling UV distribution |
| Preprocessing | Data generation, Box-Cox transform and normalisation |
| Flow model & Training | Coupling layer and normalizing flow architecture; training loop with train/test loss monitoring |
| Evaluation & Speed Comparison | Comparison of flow samples vs exact Monte Carlo, timing comparison between flow and ODE sampling |

## Background

The evolution equation for the probability distribution P(g, t) over couplings g at
RG scale t is:

$$ \frac{\partial P }{ \partial t} = -\sum_i \partial_{g_i} (\beta_{g_i} P)$$

where $\beta_{g_i} = \partial_t g_i$ are the beta functions. This is solved numerically by Monte Carlo sampling of the UV distribution and evolving each sample via the ODE system. The normalizing flow then learns this map in an amortized fashion, conditioned on the parameters of the UV distribution. For details see Eichhorn & Held (2025).


## Author

Michael Nee (mnee@fas.harvard.edu)

Postdoctoral Fellow, Department of Physics, Harvard University

