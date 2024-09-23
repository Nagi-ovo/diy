# Softmax in OpenAI Triton

import torch
import triton
import triton.language as tl
from pathlib import Path

def naive_softmax(x: torch.Tensor) -> torch.Tensor:
    x_max = x.max(dim=1, keepdim=True)[0]
    safe_x = x - x_max
    numerator = torch.exp(safe_x) 
    denominator = numerator.sum(dim=1, keepdim=True)
    sm_out = numerator / denominator
    return sm_out

@triton.jit
def _softmax_fwd_kernel(
    output_ptr,
    stride_output_rop,
    input_ptr,
    stride_input_row,
    num_cols,
    block_size: tl.constexpr,
):
    # setup input pointers
    row_index = tl.program_id(0)

    row_start_ptr = input_ptr + (row_index * stride_input_row)
    col_offsets = tl.arange(0, block_size)
    input_pointers = row_start_ptr + col_offsets

    row_mask = col_offsets < num_cols
    # move to SRAM
    row = tl.load(input_pointers, mask=row_mask, other=float("-inf"))

    # softmax itself
    safe_row = row - tl.max(row, axis=0)
    numerator = tl.exp(safe_row)
    denominator = tl.sum(numerator, axis=0)
    sm_out = numerator / denominator

    # write back to HBM
    output_row_ptr = output_ptr + (row_index * stride_output_rop)
    output_pointers = output_row_ptr + col_offsets
    tl.store(output_pointers, sm_out, mask=row_mask)

def softmax(x: torch.Tensor) -> torch.Tensor:
    """ Triton Implementation of Softmax, forward pass only"""
    rows, cols = x.shape
    assert x.dim() == 2, f"Expected 2D input, got {x.dim()}D input"
    block_size = triton.next_power_of_2(cols)
    num_warps = 4 # 32 threads per warp
    if block_size > 2047: # 2048 is the maximum block size
        num_warps = 8
    if block_size > 4095: # 4096
        num_warps = 16
    
    grid = (rows,)

    # allocate output buffer
    sm_out = torch.empty_like(x)
    
    _softmax_fwd_kernel[grid](
        sm_out,
        sm_out.stride(0),
        x,
        x.stride(0),
        cols,
        block_size=block_size,
        num_warps=num_warps

    )

    return sm_out
    

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],  # argument names to use as an x-axis for the plot
        x_vals=[
            128 * i for i in range(2, 100)
        ],  # different possible values for `x_name`
        line_arg='provider',  # argument name whose value corresponds to a different line in the plot
        line_vals=[
            'triton',
            'torch-native',
            'torch-jit',
        ],  # possible values for `line_arg``
        line_names=[
            "Triton",
            "Torch (native)",
            "Torch (jit)",
        ],  # label name for the lines
        styles=[('blue', '-'), ('green', '-'), ('green', '--')],  # line styles
        ylabel="GB/s",  # label name for the y-axis
        plot_name="softmax-performance",  # name for the plot. Used also as a file name for saving the plot.
        args={'M': 4096},  # values for function arguments not in `x_names` and `y_name`
    )
)
def benchmark(M, N, provider):
    x = torch.randn(M, N, device='cuda', dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch-native':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.softmax(x, axis=-1), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: softmax(x), quantiles=quantiles)
    if provider == 'torch-jit':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: naive_softmax(x), quantiles=quantiles)
    def gbps(ms):
        return 2 * x.nelement() * x.element_size() * 1e-09 / (ms * 0.001)
    return gbps(ms), gbps(max_ms), gbps(min_ms)

benchmark.run(show_plots=True, print_data=True, save_path=Path.cwd())

