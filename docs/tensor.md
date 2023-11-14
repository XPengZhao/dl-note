# Tensor Operations

## Tensor Creation

## Add a new dimension


```python

t = torch.rand((3,4))  ## shaped [3, 4]

t_new = t.unsqueeze(0) ## shaped [1, 3, 4]
t_new = t.unsqueeze(1) ## shaped [3, 1, 4]
t_new = t.unsqueeze(2) ## shaped [3, 4, 1]


```

