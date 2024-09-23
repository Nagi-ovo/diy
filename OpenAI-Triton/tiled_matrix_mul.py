import torch
M = 27
N = 24
K = 12
# we want M N K devisible by 3 to have even tiling
print(M, N, K)

A = torch.randn(M, K)
B = torch.randn(K, N)

output = torch.zeros(M, N)
block_M = M // 3
block_N = N // 3
block_K = K // 3
print(block_M, block_N, block_K)


total_reads = 0
total_writes = 0

for start_M in range(0, M, block_M):
    stop_M = start_M + block_M

    for start_N in range(0, N, block_N):
        stop_N = start_N + block_N

        accum = torch.zeros(block_M, block_N)
        for start_K in range(0, K, block_K):
            stop_K = start_K + block_K

            # 读取操作
            tileA = A[start_M:stop_M, start_K:stop_K]
            tileB = B[start_K:stop_K, start_N:stop_N]
            total_reads += tileA.numel() + tileB.numel()
            accum += tileA @ tileB
        output[start_M:stop_M, start_N:stop_N] = accum
        total_writes += accum.numel()
    
torch.allclose(output, A @ B)

print(total_reads, total_writes)