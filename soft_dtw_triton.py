import torch
import triton
import triton.language as tl
from euclidean_dist_triton import _euclidean_dist_triton
from itertools import product

# ----------------------------------------------------------------------------------------------------------------------

BLOCK_BATCH_R = [1, 2, 4, 8]
NUM_WARPS_R = [4, 8, 16, 32]
configs_R = [triton.Config({"BLOCK_BATCH": bb}, num_warps=nw) for bb, nw in product(BLOCK_BATCH_R, NUM_WARPS_R)]


@triton.autotune(
    configs=list(configs_R),
    key=["batch", "max_i", "max_j", "BLOCK_SEQ"],
    restore_value=["D_ptr", "R_ptr"],
)
@triton.jit
def softdtw_forward_R(
    D_ptr,
    R_ptr,
    gamma,
    bandwidth,
    batch,
    max_i,
    max_j,
    stride_b,
    stride_n,
    bulk_size: tl.constexpr,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_BATCH: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    inv_gamma = 1.0 / gamma
    offs_b = pid_batch * BLOCK_BATCH + tl.arange(0, BLOCK_BATCH)
    D_batch = D_ptr + offs_b[:, None] * stride_b
    R_batch = R_ptr + offs_b[:, None] * stride_b
    thread_id = tl.arange(0, BLOCK_SEQ)
    block_id = tl.ravel(tl.arange(0, bulk_size)[:, None] * stride_n + tl.arange(0, bulk_size)[None, :])[:, None]
    valid_batch = offs_b < batch
    # --- Forward pass: compute R ---
    num_passes = max_i + max_j - 1
    for p in tl.range(num_passes, num_stages=4):
        i_start = tl.maximum(0, p - (max_j - 1))
        i_end = tl.minimum(p, max_i - 1)
        # Apply Sakoe-Chiba bandwidth
        band_start = tl.where(bandwidth > 0, (p - bandwidth + 1) // 2, i_start)
        band_end = tl.where(bandwidth > 0, (p + bandwidth) // 2, i_end)
        i_start = tl.maximum(i_start, band_start)
        i_end = tl.minimum(i_end, band_end)

        num_valid = tl.maximum(0, i_end - i_start + 1)
        if num_valid > 0:
            valid_mask = thread_id < num_valid
            # Compute i and j for valid elements
            i = i_start + thread_id
            j = p - i
            # Boundary checks (i < max_i, j < max_j, etc.)
            valid = (i < max_i) & (j < max_j) & (i >= 0) & (j >= 0)
            block_mask = valid_batch[:, None, None] & valid_mask[None, None, :] & valid[None, None, :]
            block_idx = i * stride_n + j
            R_val = -tl.load(R_batch.expand_dims(2) + block_id + block_idx, mask=block_mask, other=float("inf")) * inv_gamma
            # Compute softmin
            R_max = tl.max(R_val, axis=1)
            R_sum = tl.sum(tl.exp(R_val - R_max.expand_dims(1)), axis=1)
            softmin = -gamma * (tl.log(R_sum) + R_max)

            # Update R
            valid_mask = valid_batch[:, None] & valid_mask[None, :] & valid[None, :]
            R_idx = (i + 1) * stride_n + j + 1
            D_val = tl.load(D_batch + R_idx, mask=valid_mask, other=0.0)
            R_new = D_val + softmin
            tl.store(R_batch + R_idx, R_new, mask=valid_mask)
        tl.debug_barrier()


BLOCK_BATCH_F = [4, 8, 16, 32]
BLOCK_ROW_F = [4, 8, 16, 32]
BLOCK_COL_F = [4, 8, 16, 32]
configs_boundary = [triton.Config({"BLOCK_BATCH": bb, "BLOCK_ROW": br, "BLOCK_COL": bc}) for bb, bc, br in product(BLOCK_BATCH_F, BLOCK_ROW_F, BLOCK_COL_F)]


@triton.autotune(
    configs=list(configs_boundary),
    key=["batch", "max_i", "max_j"],
)
@triton.jit
def softdtw_forward_boundary(
    R_ptr,
    batch,
    max_i,
    max_j,
    stride_b,
    stride_n,
    BLOCK_BATCH: tl.constexpr,
    BLOCK_ROW: tl.constexpr,
    BLOCK_COL: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_row = tl.program_id(1)
    pid_col = tl.program_id(2)

    offs_b = pid_batch * BLOCK_BATCH + tl.arange(0, BLOCK_BATCH)
    offs_i = pid_row * BLOCK_ROW + tl.arange(0, BLOCK_ROW) + 1
    offs_j = pid_col * BLOCK_COL + tl.arange(0, BLOCK_COL) + 1

    valid_batch = offs_b < batch
    valid_i = offs_i < max_i + 2
    valid_j = offs_j < max_j + 2
    valid_mask = valid_batch[:, None, None] & valid_i[None, :, None] & valid_j[None, None, :]
    grid = offs_b[:, None, None] * stride_b + offs_i[None, :, None] * stride_n + offs_j[None, None, :]
    R_val = tl.load(R_ptr + grid, mask=valid_mask, other=0.0)
    R_neginf = tl.where(R_val == float("inf"), -float("inf"), R_val)
    tl.store(R_ptr + grid, R_neginf, mask=valid_mask)


BLOCK_BATCH_E = [1, 2, 4, 8]
NUM_WARPS_E = [4, 8, 16, 32]
configs_E = [triton.Config({"BLOCK_BATCH": bb}, num_warps=nw) for bb, nw in product(BLOCK_BATCH_E, NUM_WARPS_E)]


@triton.autotune(
    configs=list(configs_E),
    key=["batch", "max_i", "max_j", "BLOCK_SEQ"],
    restore_value=["R_ptr", "D_ptr", "E_ptr"],
)
@triton.jit
def softdtw_forward_E(
    D_ptr,
    R_ptr,
    E_ptr,
    gamma,
    bandwidth,
    batch,
    max_i,
    max_j,
    stride_b,
    stride_n,
    bulk_size: tl.constexpr,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_BATCH: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    inv_gamma = 1.0 / gamma
    offs_b = pid_batch * BLOCK_BATCH + tl.arange(0, BLOCK_BATCH)
    D_batch = D_ptr + offs_b[:, None] * stride_b
    R_batch = R_ptr + offs_b[:, None] * stride_b
    E_batch = E_ptr + offs_b[:, None] * stride_b
    thread_id = tl.arange(0, BLOCK_SEQ)
    block_id = tl.ravel(tl.arange(0, bulk_size)[:, None] * stride_n + tl.arange(0, bulk_size)[None, :])[:, None]
    curr_idx = tl.full([BLOCK_BATCH, 1, BLOCK_SEQ], 0, dtype=tl.int32)
    valid_batch = offs_b < batch
    # --- Forward pass: compute E ---
    num_passes = max_i + max_j - 1
    for p in tl.range(num_passes, num_stages=4):
        rev_p = num_passes - p - 1
        i_start = tl.maximum(0, rev_p - (max_j - 1))
        i_end = tl.minimum(rev_p, max_i - 1)
        # Apply Sakoe-Chiba bandwidth in reverse pass
        band_start = tl.where(bandwidth > 0, (rev_p - bandwidth + 1) // 2, i_start)
        band_end = tl.where(bandwidth > 0, (rev_p + bandwidth) // 2, i_end)
        i_start = tl.maximum(i_start, band_start)
        i_end = tl.minimum(i_end, band_end)

        num_valid = tl.maximum(0, i_end - i_start + 1)
        if num_valid > 0:
            valid_mask = thread_id < num_valid

            i = i_start + thread_id
            j = rev_p - i

            valid = (i < max_i) & (j < max_j) & (i >= 0) & (j >= 0)
            block_mask = valid_batch[:, None, None] & valid_mask[None, None, :] & valid[None, None, :]  # for 2x2 bulk, the first element must be masked
            block_idx = (i + 1) * stride_n + j + 1

            R_val = tl.load(R_batch.expand_dims(2) + block_id + block_idx, mask=block_mask, other=-float("inf"))
            D_val = tl.load(D_batch.expand_dims(2) + block_id + block_idx, mask=block_mask, other=0.0)
            R_curr = tl.gather(R_val, curr_idx, 1)

            abc = tl.exp((R_val - R_curr - D_val) * inv_gamma)  # middle one should be R_curr
            E_val = tl.load(E_batch.expand_dims(2) + block_id + block_idx, mask=block_mask, other=0.0)

            E_new = tl.sum(abc * E_val, axis=1)
            valid_mask = valid_batch[:, None] & valid_mask[None, :] & valid[None, :]
            tl.store(E_batch + block_idx, E_new, mask=valid_mask)
        tl.debug_barrier()

# ----------------------------------------------------------------------------------------------------------------------
class _SoftDTWTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, D, gamma, bandwidth):
        dev = D.device
        dtype = D.dtype
        B, N, M = D.shape
        ctx.gamma = gamma
        ctx.bandwidth = bandwidth
        if bandwidth == 0:
            ctx.BLOCK_SEQ = min(triton.next_power_of_2(max(N + 2, M + 2)), 1024)
        else:
            ctx.BLOCK_SEQ = min(triton.next_power_of_2(max(N + 2, M + 2)), triton.next_power_of_2(bandwidth + 1))  # Prepare the arrays

        D_ = torch.zeros((B, N + 2, M + 2), device=dev, dtype=dtype)
        D_[:, 1 : N + 1, 1 : M + 1] = D
        R = torch.ones((B, N + 2, M + 2), device=dev, dtype=dtype) * float("inf")
        R[:, 0, 0] = 0
        E = torch.zeros((B, N + 2, M + 2), device=dev, dtype=dtype)
        E[:, -1, -1] = 1
        grid_R = lambda meta: (triton.cdiv(B, meta["BLOCK_BATCH"]),)
        softdtw_forward_R[grid_R](
            D_,
            R,
            gamma,
            bandwidth,
            B,
            N,
            M,
            R.stride(0),
            R.stride(1),
            bulk_size=2,
            BLOCK_SEQ=ctx.BLOCK_SEQ,
        )
        grid_boundary = lambda meta: (triton.cdiv(B, meta["BLOCK_BATCH"]), triton.cdiv(N + 2, meta["BLOCK_ROW"]), triton.cdiv(M + 2, meta["BLOCK_COL"]))
        softdtw_forward_boundary[grid_boundary](
            R,
            B,
            N,
            M,
            R.stride(0),
            R.stride(1),
        )
        R[:, N + 1, M + 1] = R[:, N, M]
        grid_E = lambda meta: (triton.cdiv(B, meta["BLOCK_BATCH"]),)
        softdtw_forward_E[grid_E](
            D_,
            R,
            E,
            gamma,
            bandwidth,
            B,
            N,
            M,
            R.stride(0),  # stride_b
            R.stride(1),  # stride_n
            bulk_size=2,
            BLOCK_SEQ=ctx.BLOCK_SEQ,
        )
        ctx.save_for_backward(R, D_, E)
        # print(R[:, 1 : N + 1, 1 : M + 1])
        # print(E[:, 1 : N + 1, 1 : M + 1])
        return E[:, 1 : N + 1, 1 : M + 1]

class SoftDTWTriton(torch.nn.Module):
    """
    The soft DTW implementation that optionally supports CUDA
    """

    def __init__(self, gamma=1.0):
        """
        Initializes a new instance using the supplied parameters
        :param gamma: sDTW's gamma parameter
        :param dist_func: Optional point-wise distance function to use. If 'None', then a default Euclidean distance function will be used.
        """
        super(SoftDTWTriton, self).__init__()
        self.gamma = gamma
        self.func_dtw = _SoftDTWTriton.apply
        self.func_dist = _euclidean_dist_triton.apply

    def forward(self, X, Y, bandwidth=0):
        """
        Compute the soft-DTW value between X and Y
        :param X: One batch of examples, batch_size x seq_len x dims
        :param Y: The other batch of examples, batch_size x seq_len x dims
        :return: The computed results
        """
        # Check the inputs and get the correct implementation
        bx, lx, dx = X.shape
        by, ly, dy = Y.shape
        # Make sure the dimensions match
        assert bx == by  # Equal batch sizes
        assert dx == dy  # Equal feature dimensions

        assert lx < 1024 or ly < 1024  # We should be able to spawn enough threads in CUDA
        # D_xy = self._euclidean_dist_func(X, Y)
        D_xy = self.func_dist(X, Y, bandwidth)
        return self.func_dtw(D_xy, self.gamma, bandwidth)