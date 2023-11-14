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