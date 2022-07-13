import torch
import triton
import triton.language as tl
import math

# WHY is this 500x slower? https://github.com/openai/triton/issues/416
# Our main bottleneck is kernel launch, and is much slower at triton v1

@triton.jit
def _forward_kernel(c_ptr, s_ptr, u_ptr, col_stride, row_stride, **meta):
    n, n_tilde, dead_index, d_max, tournament_step, BLOCK_SIZE = meta["N"], meta["N_TILDE"], meta["DEAD_INDEX"], meta["D_MAX"], meta["STEP"], meta["BLOCK_SIZE"]

    pid_x = tl.program_id(axis=0)
    temp = n_tilde - 1

    i = pid_x + tournament_step
    if pid_x == 0: i = 0
    if i >= n_tilde: i -= temp

    j = temp  - pid_x + tournament_step
    if j >= n_tilde: j -= temp

    if (i > j): 
        i,j = j, i

    if (j == dead_index) | ((j > d_max) & (i > d_max)):
        return

    theta_offset = i*n - (i+2)*(i+1)//2 + j
    c = tl.load(c_ptr+ theta_offset)
    s = tl.load(s_ptr+ theta_offset)

    offsets =  (tl.program_id(axis=1) * BLOCK_SIZE) + tl.arange(0, BLOCK_SIZE) 

    output_offsets_i =  (i * row_stride) + offsets * col_stride
    output_offsets_j = (j * row_stride) + offsets * col_stride

    maximum = n * row_stride + (n * col_stride)
    maski = output_offsets_i < maximum
    maskj = output_offsets_j < maximum

    ui = tl.load(u_ptr + output_offsets_i, mask=maski)
    uj = tl.load(u_ptr + output_offsets_j, mask=maskj)

    ioutput= (ui * c) - (uj * s)
    joutput = (uj * c) + (ui * s)

    ui = tl.store(u_ptr + output_offsets_i, ioutput, mask=maski)
    uj = tl.store(u_ptr + output_offsets_j, joutput, mask=maskj)

def _get_rotmat_constants(theta_count, n):
    d_max = n-1
    maxPairs = n*(n-1)/2

    if (theta_count < maxPairs):
        k = int(1 + math.sqrt(1 - 4*(2*theta_count - n*(n-1))))//2
        d_max -= k

    dead_index = -1
    n_tilde = n
    if n % 2 != 0:
        dead_index = n_tilde
        n_tilde += 1

    return n_tilde, d_max, dead_index

class rotMatTriton:

    @staticmethod
    def forward(thetas : torch.Tensor, n: int):
        n_tilde, d_max, dead_index = _get_rotmat_constants(thetas.size(0), n)

        C = torch.cos(thetas.detach())
        S = torch.sin(thetas.detach())
        U = torch.eye(n,n, dtype=thetas.dtype, device=thetas.device).T

        THREADS_PER_BLOCK=1024
        n_blocks_x = int(n_tilde / 2)
        n_blocks_y = triton.cdiv(n, THREADS_PER_BLOCK)
        grid = lambda meta: (n_blocks_x, n_blocks_y,)

        for tournament_step in range(n_tilde-2, -1, -1):
            _forward_kernel[grid](
                C, S, U,
                U.stride(1), U.stride(0),
                BLOCK_SIZE=THREADS_PER_BLOCK,
                N=n,
                N_TILDE=n_tilde,
                DEAD_INDEX=dead_index,
                D_MAX=d_max,
                STEP=tournament_step)

        return U