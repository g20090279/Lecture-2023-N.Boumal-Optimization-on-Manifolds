function [fOpt,iterX,iterY] = RgdProdSphere(M, x0, y0, alpha, maxIter)
%% Riemannian Gradient Descent on Products of Spheres.
% Input(s)
%   - M [matrix of size m-by-n]
%   - x0 [vector of size n]: starting point on unit sphere manifold S^(n-1)
%   - y0 [vector of size m]: starting point on unit sphere manifold S^(m-1)
%   - alpha [scalar]: fixed step size
%   - maxIter [integer]: maximum number of itrations
% Output(s)
%   - fOpt [scalar]: the maximum value of f
%   - iterX [matrix of size n-by-maxIter]: each column of it is x at the
%     corresponding iteration. There are either maxIter- or numIter-column
%     valid data
%   - iterY [matrix of size m-by-maxIter]: each column of it is y at the
%     corresponding iteration. There are either maxIter- or numIter-column
%     valid data

% Initalization
if nargin<5, maxIter = 1e4; end
lenX = length(x0);
lenY = length(y0);
iterX = zeros(lenX, maxIter);
iterY = zeros(lenY, maxIter);
eps = 1e-3;  % threshold for gradient difference of stopping algorithm

% Step (1): give initial x, y values and 
x = x0;
y = y0;
iterX(:, 1) = x;
iterY(:, 1) = y;

% Step (1): compute initial Riemannian gradient
% gradf(x,y)=( -(I-xx')My, -(I-yy')M'x )
gradf_x = -(eye(lenX)-x*x')*M*y;
gradf_y = -(eye(lenY)-y*y')*M'*x;
normGrad = norm(gradf_x) + norm(gradf_y);

% Step (2): compute the norm of the Riemannian gradient

% RGD loop
numIter = 0;
while normGrad > eps  % if norm of gradient is not approaching zero
    % Check if beyond maximum iteration
    numIter = numIter+1;
    if numIter>maxIter
        break;
    end

    % Step (3): use retraction to obtain a new point on manifold
    % Note: the retraction is just a simple norm.
    vx = - alpha * gradf_x;
    vy = - alpha * gradf_y;
    x = (x+vx)/norm(x+vx);
    y = (y+vy)/norm(y+vy);
    iterX(:, numIter) = x;
    iterY(:, numIter) = y;
    fOpt = x'*M*y;

    % Step (1)
    gradf_x = -(eye(lenX)-x*x')*M*y;
    gradf_y = -(eye(lenY)-y*y')*M'*x;
    normGrad = norm(gradf_x) + norm(gradf_y);
end
end