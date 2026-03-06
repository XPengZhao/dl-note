# 张量操作

## 张量创建

## 维度操作

### 增加一个新维度

**unsqueeze()**

```python

t = torch.rand((3,4))  ## shaped [3, 4]

t_new = t.unsqueeze(0) ## shaped [1, 3, 4]
t_new = t.unsqueeze(1) ## shaped [3, 1, 4]
t_new = t.unsqueeze(2) ## shaped [3, 4, 1]

```

---

**view()**

```python
t = torch.rand(2)  ## shaped [2]

t_new = t.view(1, 1, 2) ## shaped [1, 1, 2]

```

## 激活函数

### Sigmoid

[torch.sigmoid](https://pytorch.org/docs/stable/generated/torch.sigmoid.html)

计算输入张量每个元素的 logistic sigmoid 函数：

$$
\text{out}_i = \frac{1}{1 + e^{-\text{input}_i}}
$$

```python
t = torch.randn(4)
torch.sigmoid(t)
```

### log_softmax

$$
\text{log\_softmax}(x_i) = \log \left( \frac{e^{x_i}}{\sum_{j} e^{x_j}} \right) = x_i - \log\left(\sum_{j} e^{x_j}\right)
$$

```
import torch
```

```python
def log_softmax(input: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Compute log_softmax

    Args:
        input: Input tensor
        dim: Dimension along which to compute log_softmax
             (only -1 or last dim supported)
    Returns:
        Tensor with log_softmax applied along the specified dimension
    """
    if dim != -1 and dim != input.ndim - 1:
        raise ValueError(
            "This implementation only supports log_softmax along the last dimension"
        )

    # Flatten all dimensions except the last one
    original_shape = input.shape
    input_2d = input.reshape(-1, input.shape[-1])
    input_2d = input_2d.contiguous()

    n_rows, n_cols = input_2d.shape

    # calculate max per row for numerical stability
    max_per_row = torch.max(input_2d, dim=-1, keepdim=True).values
    input_stable = input_2d - max_per_row
    exp_input = torch.exp(input_stable)
    sum_exp = torch.sum(exp_input, dim=-1, keepdim=True)
    log_sum_exp = torch.log(sum_exp)
    log_softmax_2d = input_stable - log_sum_exp - max_per_row
    log_softmax_output = log_softmax_2d.reshape(original_shape)
    return log_softmax_output
```

为保证数值稳定性，先减去输入中的最大值：

$$
x'_i = x_i - \max_j(x_j)
$$

再计算指数和：

$$
\text{sum\_exp} = \sum_j e^{x'_j} = \sum_j e^{x_j} \cdot e^{-\max_j(x_j)}
$$

$$
\text{log\_sum\_exp} = \log(\text{sum\_exp}) = \log\left(\sum_j e^{x_j}\right) - \max_j(x_j)
$$

最后得到稳定形式的 log-softmax：

$$
\text{log\_softmax}(x_i) = x'_i - \text{log\_sum\_exp} = x_i - \log\left(\sum_j e^{x_j}\right)
$$

## CUDAGraph

为什么要使用静态张量，而不是动态张量？

使用 CUDA Graph 时，图中涉及的操作与内存访问必须保持静态，才能保证正确执行。动态张量在运行时可能改变 shape 或 size，这会破坏图捕获时假设的固定执行路径。CUDA Graph 捕获的是一段固定的操作序列和内存布局，因此张量维度变化会造成不一致甚至运行错误。所以在 CUDA Graph 中通常需要固定 shape 的静态张量。

为什么需要 buffer tensor？

buffer tensor 用于给执行过程中变化的数据提供预分配内存。由于 CUDA Graph 追求运行时不进行动态分配，buffer 可以作为占位区域承载输入或中间结果的变化，同时不改变图结构。这使得在满足 CUDA Graph 静态约束的前提下，仍能保留一定的数据灵活性，并降低运行时分配带来的开销。
