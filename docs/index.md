# GPU Configuration



## Specific GPUs Explicitly

```python
torch.cuda.set_device(0,1)

## or you can set the env variable
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

```

or set the env variable when running the code:

```bash
CUDA_VISIBLE_DEVICES=0,1 python script.py
```



**Comparison:**

If you want to restrict which GPUs are available to your script before it even starts, `CUDA_VISIBLE_DEVICES` is the way to go.

If you want to change the active GPU dynamically based on some logic in your code, you'll need to use `torch.cuda.set_device()`.

