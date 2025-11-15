# Hardware

## GPU, NPU, GPGPU

- **GPU (Graphics Processing Unit)**: A massively parallel, general-purpose processor (SIMT/SIMD) built for high throughput across many workloads—graphics, linear algebra, and most ML. Very flexible and programmable (CUDA/ROCm).

- **NPU (Neural Processing Unit)**: A domain-specific accelerator optimized for neural-net kernels (matmul/conv, activation, pooling). Prioritizes energy efficiency and latency with fixed-function or semi-programmable dataflows.

- **GPGPU (General-Purpose computing on Graphics Processing Units)**: It means using a GPU’s massively parallel hardware to run non-graphics workloads (science/ML/signal processing, etc.). Instead of drawing triangles, you launch compute kernels that apply the same operation to many data elements at once (SIMT/SIMD). This gives huge throughput for data-parallel problems.

## HBM, GDDR

GDDR (Graphics Double Data Rate) and HBM (High Bandwidth Memory) are two distinct types of memory technologies used in GPUs and high-performance computing systems, differing mainly in architecture, bandwidth, and power efficiency. GDDR uses a traditional planar design with memory chips arranged around the GPU, communicating over a relatively narrow bus at very high clock speeds to achieve high bandwidth. In contrast, HBM stacks multiple DRAM dies vertically and connects them to the processor using through-silicon vias (TSVs) and an interposer, creating an extremely wide memory interface that enables much higher bandwidth at lower clock speeds and reduced power consumption. As a result, HBM provides superior performance per watt and is often used in data centers and AI accelerators, while GDDR remains the preferred choice for consumer GPUs due to its lower cost and simpler integration.


## GPU Configuration

### Specific GPUs Explicitly

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
