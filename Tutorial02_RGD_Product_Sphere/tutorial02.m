% Tutorial 02: Rimannian gradient descent on products of unit spheres
m = 4;
n = 3;
M = rand(m,n);

% Analytical optimum of f is the largest singular value
fOptSvd = max(svd(M))

% RGD
x0 = rand(m,1);
x0 = x0/norm(x0);
y0 = rand(n,1);
y0 = y0/norm(y0);
fOptRgd = RgdProdSphere(M,x0,y0,0.1)
