# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 19:51:37 2021

@author: jaden
"""

import numpy as np, scipy as sp
from numpy import linalg as la

#needed this for SVD Hessian
def boxProduct(A,B):
  m1,n1 = A.shape
  m2,n2 = B.shape
  p = np.zeros((m1*m2,n1*n2))
  for i in range(m1):
    for j in range(m2):
      for l in range(n1):
        for k in range(n2):
          p[i*m2+j,k*n1+l] = A[i,l]*B[j,k]
  return p

#create a random matrix
max_dim = 10
m = np.maximum(np.random.randint(max_dim),2)
n = np.maximum(np.random.randint(max_dim),2)
#rank
r = np.minimum(m,n)
A = np.random.randn(m,r)@np.random.randn(r,n)

#vectorize matrix
vec = lambda X: np.reshape(X,(-1,1))
# matricize u and v, where x = [u;v]
U = lambda x: np.reshape(x[:m*r],(m,r))
V = lambda x:  np.reshape(x[m*r:],(n,r))
g = lambda x:  U(x)@V(x).T-A
#objective
f = lambda x:  .5*la.norm(g(x),'fro')**2
grad = lambda x:  np.squeeze(np.vstack((vec(g(x)@V(x)),vec(g(x).T@U(x)))))
Huu = lambda x0:  np.kron(V(x0).T@V(x0),np.eye(m))
Huv = lambda x0:  np.kron(V(x0).T,U(x0))@boxProduct(np.eye(n),np.eye(r)) + np.kron(np.eye(r),g(x0))
Hvu = lambda x0:  np.kron(U(x0).T,V(x0))@boxProduct(np.eye(m),np.eye(r)) + np.kron(np.eye(r),g(x0).T)
Hvv = lambda x0:  np.kron(U(x0).T@U(x0),np.eye(n))
#Hessian of SVD
H = lambda x0:  np.block([[Huu(x0),Huv(x0)],[Hvu(x0),Hvv(x0)]])
# this is the matvec function where H(x0) is applied to x, i.e. hess(x,x0) = H(x0)*x
Hp = lambda x0,x:  np.squeeze(np.vstack((vec(U(x)@V(x0).T@V(x0))+vec(U(x0)@V(x).T@V(x0)+g(x0)@V(x)),vec((U(x)@V(x0).T).T@U(x0)+g(x0).T@U(x))+vec((U(x0)@V(x).T).T@U(x0)))))
#random initial point
x0 = np.random.randn(r*(m+n),1)

# x1 = trustRegion(f,grad,H,x0,maxIts=100)