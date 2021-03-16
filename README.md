# Approximate Hessian Based ARC for Non-Convex Optimization
Course project for APPM 5630 - Advanced Convex Optimization, Spring 2021.

We mainly follow the ideas presented in the following papers:
- [Newton-Type Methods for Non-Convex Optimization Under Inexact Hessian Information](https://arxiv.org/abs/1708.07164)
- [Second-Order Optimization for Non-Convex Machine Learning: An Empirical Study](https://arxiv.org/abs/1708.07827)
- [Inexact Non-Convex Newton-Type Methods](https://arxiv.org/abs/1802.06925)

These are some other potentially useful papers related to the topic.
- [Newton Sketch and Sub-Sampled Newton Methods](https://arxiv.org/abs/1705.06211) (gist: subsampling is better)
- [PyHessian -- for Neural Networks](https://arxiv.org/abs/1912.07145)
- [Adaptive Cubic Regularization Methods with Inexact Hessian](https://arxiv.org/abs/1808.06239) (This is the "adaptive" adaptive regularization for cubics, worth reading since it contains a lot of implementation details, although no code since it was written in Fortran...)

I think this is the paper that proposes the adaptive variant.
- [Adaptive Cubic Regularization Algorithm](https://people.maths.ox.ac.uk/cartis/papers/cgt32.pdf)
- Here are the published version of the original ARC paper [Adaptive Cubic Regularization Algorithm Part 1](https://link.springer.com/content/pdf/10.1007/s10107-009-0286-5.pdf) [Part 2](https://link.springer.com/content/pdf/10.1007/s10107-009-0337-y.pdf)

## Abstract
We...

## Project Members
- Cooper Simpson
- Jaden Wang

## Approximate Timeline and Roadmap
We have until April 23 to get everything done.

1. 03/06 - 03/13:
  - Implement TR (Jaden)
  - Implement ARC (Cooper)
2. 03/13 - 03/20:
  - Finish testing ARC (Cooper)
  - Add pre-existing ARC implementation (Jaden): https://github.com/dalab/subsampled_cubic_regularization/blob/master/scr.py existing implementation of full Hessian exists, porting my implementation to python might not be worth the time right now, I will focus on understanding sub-smapling first.
  - Write hessian sub-sampling code (Both)
  - Start on logistic regression spambase problem (Jaden)
  - Start on shallow NN spambase problem (Cooper)

### Other plans
- Further understand and explicitly describe the conditions and requirements on the objective function for this method.
- If possible expand upon the types of problems that can be exploited in this method (e.g. finite sum minimization).
- Other ways to approximate the Hessian?

## Deliverables
- 4-6 page paper including figures. Specifically, a 1 page background, one-page method, 2-3 pages of results, and a 1/2 page conclusion.
- 10 minute presentation.

## Repository Structure
- *core*: Contains all optimization algorithm details
  - *hessian.py*: Methods for sub-sampling hessian
  - *pytorch.py*: Interaction with pytorch for use in neural networks
  - *arc*: Adaptive Regularization with Cubics algorithm implementation
- *problems*: Various target problems for applying our methods
  - *svd.py*: Compute SVD factorization using non-convex optimization
  - *spambase*: Spambase classification task

## Requirements
- numpy
- pytorch
- scipy

## Other Resources
- [Numerical Optimization by Nocedal and Wright](https://link.springer.com/book/10.1007%2F978-0-387-40065-5)
- [Second-Order Optimization Git Repo](https://github.com/git-xp/Non-Convex-Newton)
- [Inexact Newton-Type Methods Git Rrepo](https://github.com/yaozhewei/Inexact_Newton_Method)
- [Jadens Original ARC Implementation](https://github.com/tholdem/MatrixMultiplication/blob/master/CubicRegularization/cubicReg.m)
- [Somebodys Git Repo for ARC](https://github.com/cjones6/cubic_reg)
- [PyTorch Optimizer Documentation](https://pytorch.org/docs/stable/optim.html)
