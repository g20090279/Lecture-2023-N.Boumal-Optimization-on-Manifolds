# N.Boumal's Lecture: Optimization on Manifolds

This [website](https://www.nicolasboumal.net/book/index.html) links you to the relevant book and lectures provided by N. Boumal.

## Tutorial 01: Rayleigh Quotient

The problem of Tutorial 01 is the following:

--- *Problem 01 - Rayleigh Quotient* ---

Given a symmetric $A\in\mathbb{R}^{n\times n}$, we want:

$$\max_{x\in\mathbb{R}^n}x^TAx,\quad \text{s.t.}\quad \|x\|=1.$$
 
The manifold is the sphere: $\mathcal{M}=\mathbb{S}^{n-1}=\{x\in\mathbb{R}^n:\|x\|=1\}$, and the cost function to minimize is $f(x)=-x^TAx$.

--- End ---

## Tutorial 02: Riemannian Gradient Descent (RGD) on Product of Spheres

The problem of Tutorial 02 is the following:

--- *Problem 02 - Riemannian Gradient Descent on Product of Spheres* ---

Let $\mathcal{M}=\mathbb{S}^{m-1}\times\mathbb{S}^{n-1}$, which is an embedded submanifold of $\mathcal{E}=\mathbb{R}^m\times\mathbb{R}^n$ with its usual Euclidean structure. We turn $\mathcal{M}$ into a Riemannian submanifold by using the Euclidean structure of the ambient space $\mathcal{E}=\mathbb{R}^m\times\mathbb{R}^n$. Let $\mathcal{M}\in\mathbb{R}^{m\times n}$ and

$$f:\mathcal{M}\rightarrow\mathbb{R},\quad f(x,y)=x^TMy.$$
 
In the exercise Product o spheres, you showed that $f:\mathcal{M}\rightarrow\mathbb{R}$ is smooth, and worked out an expression for the Riemannian gradient of $f$. Now, we want to solve
 
$$\max_{(x,y)\in\mathcal{M}}\ f(x,y).$$

--- End ---

Note that the maximum value of $f$ on $\mathcal{M}$ is the largest singular value of $M$. Since in the book and the following, the RGD iterates towards to the negative of gradient (i.e. toward the minimum of $f$), we need to transform this maximization problem to a minimization problem by adding a negative sign before the objective function.

To perform gradient descent method, we need obtain the Riemannian gradient. We introduce the steps to approach the Riemannian gradient as the followings:

1. Since $f$ is also a smooth extension to the Euclidean space, to obtain the Riemannian Gradient, we can first obtain the Euclidean gradient $\mathrm{grad}\bar{f}(x,y)$. The product manifolds gives a nice equality that $\text{T}_{(x,y)}\mathcal{M}=\text{T}_x\mathbb{S}^{m-1}\times\text{T}_y\mathbb{S}^{n-1}$. That means we can compute the tangent space at $x$ and $y$ independently. Obviously, $\bar{f}=x^TMy$ with $x,y\in\mathcal{E}$ is a smooth extension of $f$. The Euclidean gradient can be written independently as well (Ex. 3.67)
   
  $$\begin{aligned}
  \mathrm{grad}\bar{f}(x,y)&=\left(\mathrm{grad}(x\mapsto\bar{f}(x,y))(x),\mathrm{grad}(y\mapsto\bar{f}(x,y)(y))\right)\\
  &=(\partial\bar{f}(x,y)/\partial x,\partial\bar{f}(x,y)/\partial y)\\
  &=(My,M^Tx).
  \end{aligned}$$

2. Derive a formula for the orthogonal projection from $\mathcal{E}$ onto the tangent space $\text{T}_{(x,y)}\mathcal{M}$. Taking the tangent space at $x$ as an example. We know the tangent space $\mathrm{T}_x\mathbb{S}^{m-1}=\{v\ \vert\ v^Tx=0\}$. Since the tangent space is a subspace of vector space $\mathcal{E}$, any vector $u\in\mathcal{E}$ can be decomposed into two parts - one is on the tangent space and the other is orthogonal to the tangent space, i.e. $u=u_{\|}+u_{\perp}$, where $\mathrm{Proj}_x(u)=u_{\|}$ is the orthogonal projection onto $\mathrm{T}_x\mathbb{S}^{n-1}$. Note that $x$ is also orthogonal to the tangent space. It shows the fact that $u_{\perp}$ is parallel to $x$. As a result, we need only project $u$ on $x$ and then subtract it from $u$ itself, the remaining part is $u_{\|}$, the orthogonal projection of $u$ on the tangent space. We know that $u_{\perp}=\frac{u^Tx}{x^Tx}x=u^Txx$, since $x\in\mathbb{S}^{n-1}$, i.e., $x^Tx=1$. Eventually, we obtain $\mathrm{Proj}_x(u)=u-u^Txx=u-xx^Tu=(I-xx^T)u$, resulting in the orthogonal projector for the unit sphere $\mathrm{Proj}_x=I-xx^T$.

3. Project the Euclidean gradient to the tangent space, from which we can reach the Riemannian gradient
   
  $$\begin{align}
  \mathrm{grad}f(x,y)&=\mathrm{Proj}_{(x,y)}\left(\mathrm{grad}\bar{f}(x,y)\right)\\
  &=\left(\mathrm{Proj}_x\left(\mathrm{grad}(x\mapsto\bar{f}(x,y))(x)\right),\mathrm{Proj}_y\left(\mathrm{grad}(y\mapsto\bar{f}(x,y))(y)\right)\right)\\
  &=\left((I-xx^T)My,(I-yy^T)M^Tx\right).
  \end{align}$$

4. Map the new iteration back to manifold by retraction. There are many retractions for $\mathcal{M}$. One possible and the simplest retraction is to normalize the new data

  $$\mathrm{R}:T\mathcal{M}\rightarrow\mathcal{M}:((x,y),(u,v))\mapsto\left(\frac{x+u}{\|x+u\|},\frac{y+v}{\|y+v\|}\right).$$

In a conclusion, the Riemannian Gradient is given by

--- *Algorithm Begin* ---

INPUT: $(x_0,y_0)\in\mathcal{M},\epsilon>0$, step size $\alpha>0$.
 
OUTPUT: Final position $(x,y)\in\mathcal{M}$.

1. Let $(x,y)=(x_0,y_0)$ and compute $\mathrm{grad}(-f(x,y))$,
2. While $\|\mathrm{grad}(-f(x,y))\|>\epsilon$,
3. Let $(x,y)=\mathrm{R}_{(x,y)}(-\alpha\cdot\mathrm{grad}(-f(x,y)))$,
4. Compute $\mathrm{grad}(-f(x,y))$,
5. End while.

--- *Algorithm End* ---
