# Neural Graphics

## Neural Radiance Fields (NeRF)

Neural Radiance Fields (NeRF) represents a scene as a continuous volumetric field, where the density $\sigma \in \mathbb{R}$ and radiance $\mathbf{c} \in \mathbb{R}^3$ at any 3D position $\mathbf{x} \in \mathbb{R}^3$ under viewing direction $\mathbf{d} \in \mathbb{R}^3$ are modeled by a multi-layer perceptron (MLP) $f_\theta : (\mathbf{x}, \mathbf{d}) \rightarrow (c, \sigma)$, with $\theta$ as learnable parameters. To render a pixel, the MLP first evaluates points sampled from the camera ray $\mathbf{r} = \mathbf{o} + t\mathbf{d}$ to get their densities and radiance, and then the color $\mathbf{C}(\mathbf{r})$ is estimated by volume rendering equation approximated using quadrature:

$$
\widehat{\mathbf{C}}(\mathbf{r} ; \sigma, \mathbf{c})=\sum_k T_i(\sigma)\left(1-\exp \left(-\sigma_i \delta_i\right)\right) \mathbf{c}_i
$$

where $\delta_i = t_{i+1} − t_i$ and $T_i(\sigma)=\exp \left(-\sum_{j<i} \sigma_j \delta_j\right)$. $\widehat{\mathbf{C}}$ conditioned on $\sigma$, $\mathbf{c}$, and $T$ is conditioned on $\sigma$ to simplify follow-up descriptions. We denote the **contribution of a point to the cumulative color** as its weight $\omega_i$:

$$
\omega_i=T_i(\sigma)\left(1-\exp \left(-\sigma_i \delta_i\right)\right)
$$

NeRF is optimized by minimizing the photometric loss:

$$
\mathcal{L}_{p m}=\|\widehat{\mathbf{C}}-\mathbf{C}\|_2
$$

### Physical meanings of the terms in the volume rendering equation

Alpha value of a point: $\alpha_i = 1-\exp \left(-\sigma_i \delta_i\right)$

Contribution of a point to the cumulative color: $\omega_i = T_i \alpha_i$

Opacity of each ray: $\tau = \sum_i \omega_i$

Depth of each ray: $d = \sum_i \omega_i t_i$, where $t_i$ is the distance between the i-th point and the camera

## Flow Matching

Let $\mathbb{R}^d$ denotes the data space with data points $x = (x^1, \cdots ,x^d) \in \mathbb{R}^d$.

> Think of $x$ as a location or sample point in the space where your data lives: If your data are 2D points, then $d=2$ and each $x$ is a point in the 2D plane; if your data are images of size 28x28, then $d=784$ and each $x$ is a vector representing pixel values.

Two important objects we use in this paper are: the probability density path $p: [0,1] \times \mathbb{R}^d \rightarrow \mathbb{R}_{>0}$ which is a time dependent probability density function, i.e., $\int p_t(x)dx = 1$, and a time-dependent vector field $v: [0,1] \times \mathbb{R}^d \rightarrow \mathbb{R}^d$.


> In mathematics, writing $f: A \rightarrow B$ means that $f$ is a function that takes inputs from set $A$ and produces outputs in set $B$. So here, $p$ is a function that takes two inputs: a time variable from the interval $t \in [0,1]$ and a position (data point) $x \in \mathbb{R}^d$, and it outputs a positive real number $\mathbb{R}_{>0}$ (the probability density at that point and time). $p(x, t)$ or equivalently $p_t(x)$ denotes the probability density of $x$ at $t$.

A vector field $v_t$ can be used to construct a time-dependent diffeomorphic map, called a flow, $\phi: [0,1] \times R_d \rightarrow R_d$, defined via the ordinary differential equation (ODE):