import torch
import triton
import triton.language as tl
from itertools import product

# ----------------------------------------------------------------------------------------------------------------------
BLOCK_forward_B = [4, 8, 16, 32]
BLOCK_forward_X = [4, 8, 16, 32]
BLOCK_forward_Y = [4, 8, 16, 32]
configs_dist = [
    triton.Config({"BLOCK_BATCH": bb, "BLOCK_BAND_X": bx, "BLOCK_BAND_Y": by})
    for bb, bx, by in product(
        BLOCK_forward_B,
        BLOCK_forward_X,
        BLOCK_forward_Y,
    )
]

@triton.autotune(
    configs=list(configs_dist),
    key=["B", "N", "D"],
    restore_value=["x_ptr", "y_ptr", "out_ptr"],
)
@triton.jit
def _euclid_dist_forward(
    x_ptr,
    y_ptr,
    out_ptr,
    B,
    N,
    D,
    BAND: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
    BLOCK_BATCH: tl.constexpr,
    BLOCK_BAND_X: tl.constexpr,
    BLOCK_BAND_Y: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_x = tl.program_id(1)
    pid_y = tl.program_id(2) + tl.maximum(0, pid_x * BLOCK_BAND_X - BAND) // BLOCK_BAND_Y
    offs_b = pid_batch * BLOCK_BATCH + tl.arange(0, BLOCK_BATCH)
    offs_x = pid_x * BLOCK_BAND_X + tl.arange(0, BLOCK_BAND_X)
    offs_y = pid_y * BLOCK_BAND_Y + tl.arange(0, BLOCK_BAND_Y)
    offs_d = tl.arange(0, BLOCK_DIM)

    valid_batch = offs_b < B
    valid_x = offs_x < N
    valid_y = offs_y < N
    valid_dim = offs_d < D
    mask_x = valid_batch[:, None, None] & valid_x[None, :, None] & valid_dim[None, None, :]
    mask_y = valid_batch[:, None, None] & valid_y[None, :, None] & valid_dim[None, None, :]
    mask_band = tl.abs(offs_x[:, None] - offs_y[None, :]) <= BAND
    mask_out = valid_batch[:, None, None] & valid_x[None, :, None] & valid_y[None, None, :] & mask_band[None, :, :]

    grid_x = offs_b[:, None, None] * N * D + offs_x[None, :, None] * D + offs_d[None, None, :]
    grid_y = offs_b[:, None, None] * N * D + offs_y[None, :, None] * D + offs_d[None, None, :]
    grid_out = offs_b[:, None, None] * N * N + offs_x[None, :, None] * N + offs_y[None, None, :]

    x = tl.load(x_ptr + grid_x, mask=mask_x, other=0.0)
    y = tl.load(y_ptr + grid_y, mask=mask_y, other=0.0)
    diff = x.expand_dims(2) - y.expand_dims(1)
    out = tl.sum(diff * diff, axis=3)
    tl.store(out_ptr + grid_out, out, mask=mask_out)

class _euclidean_dist_triton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, Y, bandwidth):
        """
        Compute Euclidean distances within a Sakoe-Chiba band using a Triton kernel.
        """
        B, N, D = X.shape
        _, M, _ = Y.shape
        assert N == M

        out = torch.zeros((B, N, M), device=X.device, dtype=X.dtype)
        # ctx.save_for_backward(X, Y)
        ctx.BAND = N if bandwidth == 0 or bandwidth >= N else bandwidth + 1
        ctx.BLOCK_DIM = triton.next_power_of_2(D)

        grid = lambda meta: (
            triton.cdiv(B, meta["BLOCK_BATCH"]),
            triton.cdiv(N, meta["BLOCK_BAND_X"]),
            triton.cdiv(2 * ctx.BAND + meta["BLOCK_BAND_X"] + meta["BLOCK_BAND_Y"] if ctx.BAND != N else N, meta["BLOCK_BAND_Y"]),
        )
        _euclid_dist_forward[grid](
            X,
            Y,
            out,
            B,
            N,
            D,
            ctx.BAND,
            ctx.BLOCK_DIM,
        )
        return out


def dist_triton(x, y, bandwidth=0):
    """
    Calculates the Euclidean distance between each element in x and y per timestep
    """
    return _euclidean_dist_triton.apply(x, y, bandwidth)