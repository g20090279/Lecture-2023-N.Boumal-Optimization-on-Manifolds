% Boumal's Lectur "Optimization on Manifolds"
% Tutorial 01: Rayleigh
%
% Given a symmetric A in R of size n-by-n, we want
%     max_{x in R^n} x^T*A*x
%     s.t. ||x||=1
% The corresponding manifold is the sphere
%    M = S^{n-1} = {x in R^n : ||x||=1}
% The cost function to minimize is:
%    f(x) = -x^T*A*x

clear; clf; clc;

addpath(genpath('../Manopt_7.1/manopt'));

n = 20;
A = randn(n);
A = A+A';

problem.M = spherefactory(n);
problem.cost = @(x) -x'*A*x;

% Add gradient and Hessian to speed the algorithm
problem.egrad = @(x) -2*A*x;
problem.ehess = @(x,u) -2*A*u;

% Algorithm 1: steepest descent
% x = steepestdescent(problem);

% Algorithm 2: trust region
x = trustregions(problem);

% Compare by another analytical solution
x'*A*x, max(eig(A))

% Check the slop of gradient and Hessian
checkgradient(problem)
checkhessian(problem)