import math
import torch
from typing import List, Optional

@torch.jit.script
def compute_distances(
    pos: torch.Tensor,
    mapping: torch.Tensor,
    cell_shifts: Optional[torch.Tensor] = None,
):
    r"""Compute the distance between the positions in :obj:`pos` following the
    :obj:`mapping` assuming that mapping indices follow::

     i--j

    such that:

    .. math::

        r_{ij} = ||\mathbf{r}_j - \mathbf{r}_i||_{2}

    In the case of periodic boundary conditions, :obj:`cell_shifts` must be
    provided so that :math:`\mathbf{r}_j` can be outside of the original unit
    cell.
    """
    assert mapping.dim() == 2
    assert mapping.shape[0] == 2

    if cell_shifts is not None:
        dr = (pos[mapping[1]] + cell_shifts[:, :, 1]) - pos[mapping[0]]
    else:
        dr = pos[mapping[1]] - pos[mapping[0]]

    return dr.norm(p=2, dim=1)

def compute_cell_shifts(
    pos: torch.Tensor,
    mapping: torch.Tensor,
    pbc: torch.Tensor,
    cell: torch.Tensor,
    batch: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the minimum vector using index 0 as reference
        Scale vectors based on box size and shift if greater than half the box size
        Adopted from ase.geometry naive_find_mic
            https://gitlab.com/ase/ase/
    Inputs:
        pos: (n_coords_over_frames x 3(x,y,z))
            positions from AtomicData object
        mapping: (order_map x n_mapping)
            index mapping from AtomicData object
            order_map = 2,3,4,etc. for harmonic, angle, dihedral, etc.
        pbc: (frames x 3)
            whether to apply cell shift in this dimension
        cell: (frames x 3 x 3)
            unit cell
        batch: (n_mapping)
            which frame corresponds to each mapping
    Returns:
        cell_shifts: (n_mapping x 3(x,y,z) x order_map)
            Integer values of how many unit cells to shift for minimum image convention
                based on the first index in mapping
            First column is all zeros by convention as shift to self
    """

    # Must wrap with no grad in order to avoid error when passing through forward
    with torch.no_grad():
        mapping = mapping.T
        cell_shifts = torch.zeros(
            mapping.shape[0], 3, mapping.shape[1], dtype=pos.dtype
        ).to(pos.device)
        drs = torch.zeros(
            mapping.shape[0], 3, mapping.shape[1], dtype=pos.dtype
        ).to(pos.device)
        for ii in range(1, cell_shifts.shape[-1]):
            drs[:, :, ii] = pos[mapping.T[0]] - pos[mapping.T[ii]]
            batch_ids = batch[mapping.T[0]]
            scaled = torch.linalg.solve(
                cell[batch_ids].to(drs.dtype), drs[:, :, ii]
            )
            cell_shifts[:, :, ii] = torch.floor(scaled + 0.5)
            cell_shifts[:, :, ii] = pbc[batch_ids] * torch.einsum(
                "bij,bj->bi",
                cell[batch_ids].to(drs.dtype),
                cell_shifts[:, :, ii],
            )
    return cell_shifts