# Approximate Hessian Based ARC for Non-Convex Optimization
Course project for APPM 5630 - Advanced Convex Optimization, Spring 2021.

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
  - Add pre-existing ARC implementation (Jaden): SCR implementation of full Hessian (cholesky) ARC exists, porting my implementation to python might not be worth the time right now since we just want to use it to check but not in experiment (too slow), I will focus on understanding sub-sampling first.
  - Write hessian sub-sampling code (Both)
  - Start on logistic regression spambase problem (Jaden)

3. 03/21 - 03/28
  - Get shallow NN spambase problem working via pytorch (both)
  - Write outline for reference papers (Jaden)
  - Write the theoretic background section of the paper if time permitted (both)

4. 03/28 - 04/04
  - Get ARC working with pytorch (Cooper)
  - Finish a draft of theoretical section (Jaden)
  - Get more practice with pytorch tutorials (Jaden)
  - Start the experiment section if time permits (both)

5. 04/10 - 04/17
  - Clean up optimizer and try to improve (Cooper)
  - Implement PyTorch Lightning to use profiler (Cooper)
  - Implement MNIST dataset problem (Cooper)
  - Go through papers and continue writing (Cooper/Both)

### Other plans
- Further understand and explicitly describe the conditions and requirements on the objective function for this method.
- If possible expand upon the types of problems that can be exploited in this method (e.g. finite sum minimization).
- Other ways to approximate the Hessian, including non-uniform sampling presented by Xu 2020, dynamic Hessian accuracy by Bellavia 2019.

## Deliverables
- 4-6 page paper including figures. Specifically, a 1 page background, one-page method, 2-3 pages of results, and a 1/2 page conclusion.
- 10 minute presentation.

## Repository Structure
- *core*: Contains all optimization algorithm details
  - *hessian.py*: Methods for sub-sampling hessian
  - *pytorch.py*: Interaction with PyTorch for use in neural networks
  - *arc*: Adaptive Regularization with Cubics algorithm implementation
- *problems*: Various target problems for applying our methods
  - *svd.py*: Compute SVD factorization using non-convex optimization
  - *spambase*: Spambase classification task via standard Logistic Regression and a shallow NN

## Requirements
- numpy
- pytorch
- scipy

# Resources and References
There is a large amount of prior research on this subject that has inspired our work and provided invaluable reference. These sources have been organized below by publication date, authors, and material. Links to papers and code repositories have been provided when possible.

- Introduction of Adaptive Cubic Regularization (ARC) methods. The first two papers consider unconstrained optimization of convex problems, while the third extends the work to include non-convex problems.
  - Adaptive Cubic Regularization Methods for Unconstrained Optimization [Part 1](https://link.springer.com/content/pdf/10.1007/s10107-009-0286-5.pdf), [Part 2](https://link.springer.com/content/pdf/10.1007/s10107-009-0337-y.pdf)
  - [An Adaptive Cubic Regularization Algorithm for Non-Convex Optimization](https://people.maths.ox.ac.uk/cartis/papers/cgt32.pdf)

- As stated by the authors this appears to be the first work with strong results on sub-sampling in ARC for non-convex optimization. They also specifically consider the application to machine learning via finite-sum minimization.
  - [Sub-Sampled Cubic Regularization for Non-Convex Optimization](https://arxiv.org/abs/1705.05933)
    - [GitHub Repository](https://github.com/dalab/subsampled_cubic_regularization)

- The following three papers come from a similar set of authors and were the original inspiration for our work. All three deal with non-convex optimization through Newton-type methods (e.g. ARC) using approximate hessian information via sub-sampling. The first paper establishes the theory for their approach with the second being more of a numerical study. The third paper introduces inexact gradient information in addition to sub-sampled hessian.
  - [Newton-Type Methods for Non-Convex Optimization Under Inexact Hessian Information](https://arxiv.org/abs/1708.07164)
  - [Second-Order Optimization for Non-Convex Machine Learning: An Empirical Study](https://arxiv.org/abs/1708.07827)
    - [GitHub Repository](https://github.com/git-xp/Non-Convex-Newton)
  - [Inexact Non-Convex Newton-Type Methods](https://arxiv.org/abs/1802.06925)
    - [GitHub Repository](https://github.com/yaozhewei/Inexact_Newton_Method)

- In this paper a dynamic version of ARC is introduced where the degree of accuracy in the hessian sub-sampling changes adaptively.
  - [Adaptive Cubic Regularization Methods with Inexact Hessian](https://arxiv.org/abs/1808.06239)

- This work introduces a software tool for extracting hessian information from a neural network.
  - [PyHessian: Neural Networks Through the Lens of the Hessian](https://arxiv.org/abs/1912.07145)


## Other References
- [Numerical Optimization by Nocedal and Wright](https://link.springer.com/book/10.1007%2F978-0-387-40065-5)
- [Jadens Original ARC Implementation](https://github.com/tholdem/MatrixMultiplication/blob/master/CubicRegularization/cubicReg.m)
- [Somebodys Git Repo for ARC](https://github.com/cjones6/cubic_reg)
- [PyTorch Optimizer Documentation](https://pytorch.org/docs/stable/optim.html)
