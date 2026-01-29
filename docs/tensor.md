# Tensor Operations

## Tensor Creation

## Dimension Operations

### Add a new dimension

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

## Activate Function

### Sigmoid

[torch.sigmoid](https://pytorch.org/docs/stable/generated/torch.sigmoid.html)

Computes the logistic sigmoid function of the elements of input.

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


Compute the maximum value in the input tensor and subtract it from all elements for numerical stability.

$$
x'_i = x_i - \max_j(x_j)
$$

calculate the sum of exponentials of the values.
$$
\text{sum\_exp} = \sum_j e^{x'_j} = \sum_j e^{x_j} \cdot e^{-\max_j(x_j)}
$$

$$
\text{log\_sum\_exp} = \log(\text{sum\_exp}) = \log\left(\sum_j e^{x_j}\right) - \max_j(x_j)
$$

Finally, compute the log-softmax values using the stabilized inputs.

$$
\text{log\_softmax}(x_i) = x'_i - \text{log\_sum\_exp} = x_i - \log\left(\sum_j e^{x_j}\right)
$$

