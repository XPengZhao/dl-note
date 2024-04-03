# Tensor Operations

## Tensor Creation

## Dimension Operations

### Add a new dimension


```python

t = torch.rand((3,4))  ## shaped [3, 4]

t_new = t.unsqueeze(0) ## shaped [1, 3, 4]
t_new = t.unsqueeze(1) ## shaped [3, 1, 4]
t_new = t.unsqueeze(2) ## shaped [3, 4, 1]


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



```
