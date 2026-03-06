# 神经图形

## 神经辐射场（NeRF）

神经辐射场（NeRF）将场景表示为连续体渲染场：在任意 3D 位置 $\mathbf{x} \in \mathbb{R}^3$、视角方向 $\mathbf{d} \in \mathbb{R}^3$ 下，体密度 $\sigma \in \mathbb{R}$ 与辐射颜色 $\mathbf{c} \in \mathbb{R}^3$ 由多层感知机（MLP）$f_\theta : (\mathbf{x}, \mathbf{d}) \rightarrow (c, \sigma)$ 建模，其中 $\theta$ 为可学习参数。渲染某个像素时，先沿相机光线 $\mathbf{r} = \mathbf{o} + t\mathbf{d}$ 采样点并预测其密度与颜色，再通过体渲染方程（数值积分近似）计算像素颜色 $\mathbf{C}(\mathbf{r})$：

$$
\widehat{\mathbf{C}}(\mathbf{r} ; \sigma, \mathbf{c})=\sum_k T_i(\sigma)\left(1-\exp \left(-\sigma_i \delta_i\right)\right) \mathbf{c}_i
$$

其中 $\delta_i = t_{i+1} − t_i$，$T_i(\sigma)=\exp \left(-\sum_{j<i} \sigma_j \delta_j\right)$。后文为简化叙述，常将 $\widehat{\mathbf{C}}$ 视为由 $\sigma, \mathbf{c}, T$ 决定，其中 $T$ 又由 $\sigma$ 决定。我们把**单个采样点对累计颜色的贡献**记为权重 $\omega_i$：

$$
\omega_i=T_i(\sigma)\left(1-\exp \left(-\sigma_i \delta_i\right)\right)
$$

NeRF 通过最小化光度重建损失进行优化：

$$
\mathcal{L}_{p m}=\|\widehat{\mathbf{C}}-\mathbf{C}\|_2
$$

### 体渲染方程中各项的物理意义

点的 alpha 值：$\alpha_i = 1-\exp \left(-\sigma_i \delta_i\right)$

点对累计颜色的贡献：$\omega_i = T_i \alpha_i$

每条光线的不透明度：$\tau = \sum_i \omega_i$

每条光线的深度：$d = \sum_i \omega_i t_i$，其中 $t_i$ 是第 i 个点到相机的距离

## Flow Matching

设 $\mathbb{R}^d$ 为数据空间，数据点记为 $x = (x^1, \cdots ,x^d) \in \mathbb{R}^d$。

> 可以把 $x$ 理解为数据空间中的一个位置或样本点：如果数据是二维点，则 $d=2$，每个 $x$ 是平面上的一点；如果数据是 28x28 图像，则 $d=784$，每个 $x$ 是像素向量。

本文中两个核心对象为：概率密度路径 $p: [0,1] \times \mathbb{R}^d \rightarrow \mathbb{R}_{>0}$（随时间变化的概率密度函数，满足 $\int p_t(x)dx = 1$），以及随时间变化的向量场 $v: [0,1] \times \mathbb{R}^d \rightarrow \mathbb{R}^d$。

> 在数学中，$f: A \rightarrow B$ 表示函数 $f$ 以集合 $A$ 中元素为输入，并输出到集合 $B$。这里 $p$ 接收两个输入：时间变量 $t \in [0,1]$ 与位置（样本点）$x \in \mathbb{R}^d$，输出一个正实数 $\mathbb{R}_{>0}$（该时刻该位置的概率密度）。$p(x, t)$ 或 $p_t(x)$ 都表示时刻 $t$ 下点 $x$ 的概率密度。

向量场 $v_t$ 可定义一个随时间变化的微分同胚映射（flow）$\phi: [0,1] \times R_d \rightarrow R_d$，其由以下常微分方程（ODE）给出：
