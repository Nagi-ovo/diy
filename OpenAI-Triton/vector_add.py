import torch

import triton
import triton.language as tl

@triton.jit
def kernel_vector_addition(a_ptr, b_ptr, out_ptr,
                           num_elems: tl.constexpr,
                           block_size: tl.constexpr,):

    pid = tl.program_id(axis=0)
    # tl.device_print("pid", pid)
    block_start = pid * block_size # 0 * 2 = 0, 1 * 2 = 2,
    thread_offsets = block_start + tl.arange(0, block_size)
    mask = thread_offsets < num_elems
    a_pointers = tl.load(a_ptr + thread_offsets, mask=mask)
    b_pointers = tl.load(b_ptr + thread_offsets, mask=mask)
    res = a_pointers + b_pointers
    tl.store(out_ptr + thread_offsets, res, mask=mask)



def ceil_div(x: int,y: int) -> int:
    return (x + y - 1) // y

def vector_addition(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    output_buffer = torch.empty_like(a)
    assert a.is_cuda and b.is_cuda
    num_elems = a.numel()
    assert num_elems == b.numel() # todo - handel mismatched sizes

    block_size = 1024
    grid_size = ceil_div(num_elems, block_size)
    grid = (grid_size,)
    num_warps = 8

    k2 = kernel_vector_addition[grid](a, b, output_buffer,
                                      num_elems,
                                      block_size,
                                      num_warps=num_warps
                                      )
    return output_buffer

def verify_numerics() -> bool:
    # verify numerical fidelity
    torch.manual_seed(2020) # seed both cpu and gpu
    vec_size = 8192
    a = torch.rand(vec_size, device='cuda')
    b = torch.rand_like(a)
    torch_res = a + b
    triton_res = vector_addition(a, b)
    fidelity_correct = torch.allclose(torch_res, triton_res)
    print(f"{fidelity_correct=}")

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'], # Argument names to use as an x-axis for the plot
        x_vals=[
            2**i for i in range(10, 28)
        ], # Different possible values for `x_name`
        x_log=True, # x-axis is logarithmic
        line_arg='provider', # Argument name whose value corresponds to a different line in the plot
        line_vals=['triton', 'torch'], # Possible values for `line_arg` 
        line_names=["Triton", "Torch"], # Label name of the lines 
        styles=[('blue', '-'), ('green', '-')], # Line color and style
        ylabel='GB/s', # Label for the y-axis
        plot_name='vector-add-performance', # Name for the plot. Used also as a file name for saving the plot.
        args={}, # Values for function arguments not in `x_names` and `y_name`
    )
)

def benchmark(size, provider):
    x = torch.rand(size, device='cuda', dtype=torch.float32)
    y = torch.rand(size, device='cuda', dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_mas = triton.testing.do_bench(lambda: x + y, quantiles=quantiles)
        # for example. quantiles = 0.5, 0.2, 0.8, then return median, 20th percentile, 80th percentile
    if provider == 'triton':
        ms, min_ms, max_mas = triton.testing.do_bench(lambda: vector_addition(x, y), quantiles=quantiles)
    def gbps(ms):
        return 12 * size / ms * 1e-06
    return gbps(ms), gbps(max_mas), gbps(min_ms)

if __name__ == '__main__':
    # verify_numerics()
    benchmark.run(print_data=True, show_plots=True, save_path='./vec_add_perf')
    

