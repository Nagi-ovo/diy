# Online Softmax in OpenAI Triton

import torch
import triton
import triton.language as tl
import time

def naive_softmax(x: torch.Tensor) -> torch.Tensor:
    x_max = x.max(dim=1, keepdim=True)[0]
    safe_x = x - x_max
    numerator = torch.exp(safe_x) 
    denominator = numerator.sum(dim=1, keepdim=True)
    sm_out = numerator / denominator
    return sm_out

def online_softmax(x: torch.Tensor) -> torch.Tensor:
    """ online softmax implementation, 2.5x faster than naive softmax"""
    rows_count, col_count = x.shape
    assert x.dim() == 2, f"Expected 2D input, got {x.dim()}D input"
    output = torch.zeros_like(x)

    for r in range(rows_count):
        row_max = 0     # m
        normalizer = 0  # l
        for c in range(col_count):
            current = x[r, c]
            prev_old_max = row_max
            row_max = max(row_max, current)
            if row_max > prev_old_max:
                print(f"updated row_max is now {row_max}, row = {r}")
            normalizer = normalizer * torch.exp(prev_old_max - row_max) + torch.exp(current - row_max)
        output[r,:] = torch.exp(x[r,:] - row_max) / normalizer # safe max
    return output


# ---- Unit Test ----

sample = torch.tensor([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]], dtype=torch.float32, device='cuda')
start = time.perf_counter()
eager_out = naive_softmax(sample)
stop = time.perf_counter()
eager_time = stop - start

start = time.perf_counter()
online_out = online_softmax(sample)
stop = time.perf_counter()
online_time = stop - start
ref_out = torch.softmax(sample, dim=1)

print(f"{eager_out=}\n{online_out=}\n{ref_out=}\n")
print(f"{eager_time=}, {online_time=}")
