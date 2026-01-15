import math
import torch
import torch.nn.functional as F

def calc_psnr(test, ref):
    mse = ((test - ref) ** 2).mean([-2, -1])
    return 20 * torch.log10(ref.max() / torch.sqrt(mse)).cpu().mean().item()

def make_coord(shape, device):
    view_seqs = -1 + 1 / shape[0] + (2 / shape[0]) * (torch.arange(shape[0], device=device).float())
    det_seqs = -1 + 1 / shape[1] + (2 / shape[1]) * (torch.arange(shape[1], device=device).float())
    coord = torch.stack(torch.meshgrid((det_seqs, view_seqs), indexing="xy"), dim=-1).unsqueeze(0)  # view 에 대해서는 sampling으로 바꿔야됨
    return coord


def reshape_patch(patches, batch):
    blocks = int(math.sqrt(patches.shape[0] / batch))
    patches = patches.view(batch, blocks, blocks, patches.shape[2], patches.shape[3])
    patches = patches.permute(0, 2, 3, 1, 4).contiguous().view(batch, blocks * patches.shape[-2], blocks * patches.shape[-1])
    return patches.unsqueeze(1)



def grid_sample_slope(sinogram, coord, slope):
    # each [1, 64, 1, 768], [1, 512, 768, 2], [1, 1, 512, 768]
    batch, target_view, target_det, _ = coord.shape
    _, channel, in_view, in_det = sinogram.shape

    input_coord = make_coord((in_view, in_det), device=sinogram.device).tile(batch, 1, 1, 1)
    target_coord = make_coord((target_view, target_det), device=sinogram.device).tile(batch, 1, 1, 1)
    view_coord = make_coord((target_view, in_det), device=sinogram.device).tile(batch, 1, 1, 1)
    view_coord[:, :, :, 1] += (target_view // in_view - 1) / target_view

    up_sino, lens_view = [], []
    tot_view = 0

    for i in [-1, 1]:
        coord_ = coord.clone()
        coord_[:, :, :, 1] += i / in_view + 1e-6

        view_coord_ = view_coord.clone()
        view_coord_[:, :, :, 1] += i / in_view + 1e-6
        temp_sino = F.grid_sample(sinogram, view_coord_, mode="nearest", padding_mode="border", align_corners=False)

        temp_coord = F.grid_sample(input_coord.permute(0, 3, 1, 2), coord_, mode="nearest", padding_mode="border", align_corners=False)
        temp_coord = temp_coord.permute(0, 2, 3, 1)

        rel_view = (coord[:, :, :, 1] - temp_coord[:, :, :, 1]) * in_view
        shift_det = slope[:, 0, :, :] * rel_view / 2

        det_coord_ = target_coord.clone()
        det_coord_[:, :, :, 0] += shift_det * (2 / in_det)

        up_sino.append(
            F.grid_sample(temp_sino, det_coord_, mode="bilinear", padding_mode="border", align_corners=False).permute(0, 2, 3, 1).view(batch, -1, channel)
        )
        rel_view_abs = rel_view.view(batch, -1, 1).abs() + 1e-6
        tot_view += rel_view_abs
        lens_view.append(rel_view_abs)
    return (up_sino[0] * lens_view[1] + up_sino[1] * lens_view[0]) / tot_view