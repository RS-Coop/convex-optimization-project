# Approximate Hessian Based ARC for Non-Convex Optimization
Course project for APPM 5630 - Advanced Convex Optimization, Spring 2021.

We mainly follow the ideas presented in the following papers:
- [Newton-Type Methods for Non-Convex Optimization Under Inexact Hessian Information](https://arxiv.org/abs/1708.07164)
- [Second-Order Optimization for Non-Convex Machine Learning: An Empirical Study](https://arxiv.org/abs/1708.07827)

## Abstract
We...

## Project Members
- Cooper Simpson
- Jaden Wang

## Approximate Timeline and Roadmap
We have until April 23 to get everything done.

1. Port the optimization code from Matlab to Python and try to connect each piece with the related theory. This includes implementing ARC, solving the CR sub-problem, and approximating the Hessian.
2. Understand what needs to be done to integrate our code with PyTorch.
3. Further understand and explicitly describe the conditions and requirements on the objective function for this method.
4. If possible expand upon the types of problems that can be exploited in this method (e.g. finite sum minimization).
5. Other ways to approximate the Hessian?
6. Do experiments

# Deliverables
- 4-6 page paper including figures. Specifically, a 1 page background, one-page method, 2-3 pages of results, and a 1/2 page conclusion.
- 10 minute presentation.

# Repository Structure
- *aarc* (Approximate Adaptive Regularization with Cubics): This folder is Coopers port of the optimization code.
- *dl_experiment*: Code for applying our method to a number of deep learning models.

## Other Resources
- [Second-Order Optimization Git Repo](https://github.com/git-xp/Non-Convex-Newton)
