# Insert Cool Name Here
Course project for APPM 5630 - Advanced Convex Optimization

## Team Members
Cooper Simpson 
Jaden Wang

Project ideas:
1. 2nd order Newton methods under inexact Hessian https://link.springer.com/article/10.1007/s10107-019-01405-z

Background: I did some research last summer using a 2nd order Newton's method called adaptive regularization for cubics (ARC) with exact Hessian and found it to be a super powerful algorithm for non-convex problems, because it uses 2nd order information to avoid saddle points and converges superlinearly near the optimal solution. This paper allows us to cheaply approximate the Hessian instead (because true Hessian is a pain in the butt to find) and claims the computational cost to be comparable to SGD. If the claim is true this would be quite remarkable and we can use it to solve many hard problems where SGD fails. It would be interesting to implement this algorithm and either replicate their results or apply it to our own problems or compare it to ARC with true Hessian.

2. sparse PCA via Riemannian optimization https://epubs.siam.org/doi/pdf/10.1137/080731359?casa_token=LC-4M4WJmgAAAAAA:xk-8RmscEv8PlnZcelQcm3iHsLtg63VSigm5puKzv7TUwMEU-LbkY1AVDd6joRBP22lKoP4xsw

Background: I implemented a Riemmanian version of ARC last summer without much understanding due to lack of prereqs in differential geometry. It is very fascinating to me and I thought perhaps starting from an easier problem/manifold would yield a gentler learning curve. Also semidefinite matrix cone is pretty relevant to this class and sparse PCA is super useful in application.

3. Optimization on real-valued function with complex input using Wirtinger Calculus https://epubs.siam.org/doi/pdf/10.1137/110832124?casa_token=JZcLu_vMxCAAAAAA:C2ioOdP[…]WRJlf8pSy-W1BIWy_5U8v8l9b6Od4StnYr0zI_PK7JF3LgXCsMmuHtbAzOr6w
https://www.nature.com/articles/s41598-019-52289-0
Background: I'm currently trying to optimize such function for research, and the fact that the function involves complex steps is causing lots of trouble using traditional method. It seems like Wirtinger gradient is the proper method for this kind of functions, which appear all the time in certain applications (e.g. least squares objective function of signals involving integral transforms, quantum tomography, etc).
