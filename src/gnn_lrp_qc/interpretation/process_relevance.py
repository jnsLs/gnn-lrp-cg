import types
from typing import Callable, Dict, Optional, List
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from tqdm import tqdm

from schnetpack import properties
import schnetpack.nn.so3 as so3
from schnetpack.data.loader import _atoms_collate_fn
from gnn_lrp_qc.utils.molecular_graph import get_all_walks
from gnn_lrp_qc.utils.batch import chunker


def xai_forward(self, input: torch.Tensor):
    y = F.linear(input, self.weight, self.bias)
    y = y * (self.activation(y) / y).detach()
    return y


def apply_quotient_rule(model):
    for name, module in model.named_children():
        if isinstance(module, nn.Module):
            # If the module is a submodule, check its children recursively
            if hasattr(module, "activation") and (
                module.activation is nn.Identity or module.activation is not None
            ):
                print(
                    f"{module.__class__.__name__} {name} has non-None activation: {module.activation}"
                )
                module.forward = types.MethodType(xai_forward, module)
            apply_quotient_rule(module)


class ProcessRelevance:
    def __init__(self, model, device, target, gamma, zero_bias=False):
        self.target = target
        self.gamma = gamma
        self.device = device

        apply_quotient_rule(model)
        self.model = model
        architecture_name = model.representation.__class__.__name__

        if architecture_name == "SchNet":
            self.representation_forward = self.schnet_forward
        elif architecture_name == "PaiNN":
            self.representation_forward = self.painn_forward
        elif architecture_name == "SO3net":
            self.representation_forward = self.so3net_forward
        else:
            raise NotImplementedError(
                f"Architecture {architecture_name} not implemented."
            )

        self._zero_bias() if zero_bias else None

    def process(self, sample):
        pass

    def schnet_forward(
        self, inputs: Dict[str, torch.Tensor], walk: Optional[List] = None
    ):
        atomic_numbers = inputs[properties.Z]
        r_ij = inputs[properties.Rij]
        idx_i = inputs[properties.idx_i]
        idx_j = inputs[properties.idx_j]

        # compute atom and pair features
        x = self.model.representation.embedding(atomic_numbers)
        d_ij = torch.norm(r_ij, dim=1)
        f_ij = self.model.representation.radial_basis(d_ij)
        rcut_ij = self.model.representation.cutoff_fn(d_ij)

        # store type embedding
        x = Variable(x.data, requires_grad=True)
        h0 = x

        # do masking
        if walk is not None:
            mask = torch.zeros(x.shape).to(self.device)
            # multi-walk interpretation
            if isinstance(walk[0], tuple) or isinstance(walk[0], list) or isinstance(walk[0], np.ndarray):
                assert len(inputs[properties.n_atoms]) == len(
                    walk
                ), "Input data must contain as many collated frames as input walks"
                offset = 0
                for i, w in enumerate(walk):
                    mask[w[0] + offset] = 1
                    offset += inputs[properties.n_atoms][i]
            # single-walk interpretation
            else:
                mask[walk[0]] = 1
            x = x * mask + (1 - mask) * x.data

        # compute interaction block to update atomic embeddings
        for layer_idx, interaction in enumerate(self.model.representation.interactions):
            v = interaction(x, f_ij, idx_i, idx_j, rcut_ij)
            x = x + v

            if walk is not None:
                mask = torch.zeros(x.shape).to(self.device)
                # multi-walk interpretation
                if isinstance(walk[0], tuple) or isinstance(walk[0], list) or isinstance(walk[0], np.ndarray):
                    offset = 0
                    for i, w in enumerate(walk):
                        mask[w[layer_idx + 1] + offset] = 1
                        offset += inputs[properties.n_atoms][i]
                # single-walk interpretation
                else:
                    mask[walk[layer_idx + 1]] = 1
                x = x * mask + (1 - mask) * x.data

        inputs["scalar_representation"] = x

        return inputs, h0

    def painn_forward(
        self, inputs: Dict[str, torch.Tensor], walk: Optional[List] = None
    ):
        # get tensors from input dictionary
        atomic_numbers = inputs[properties.Z]
        r_ij = inputs[properties.Rij]
        idx_i = inputs[properties.idx_i]
        idx_j = inputs[properties.idx_j]
        n_atoms = atomic_numbers.shape[0]

        # compute atom and pair features
        d_ij = torch.norm(r_ij, dim=1, keepdim=True)
        dir_ij = r_ij / d_ij
        phi_ij = self.model.representation.radial_basis(d_ij)
        fcut = self.model.representation.cutoff_fn(d_ij)

        filters = self.model.representation.filter_net(phi_ij) * fcut[..., None]
        if self.model.representation.share_filters:
            filter_list = [filters] * self.model.representation.n_interactions
        else:
            filter_list = torch.split(
                filters, 3 * self.model.representation.n_atom_basis, dim=-1
            )

        q = self.model.representation.embedding(atomic_numbers)[:, None]
        qs = q.shape
        mu = torch.zeros((qs[0], 3, qs[2]), device=q.device)

        # store type embedding
        q = Variable(q.data, requires_grad=True)
        h0 = q

        # do masking
        if walk is not None:
            mask = torch.zeros(q.shape).to(self.device)
            # multi-walk interpretation
            if isinstance(walk[0], tuple) or isinstance(walk[0], list) or isinstance(walk[0], np.ndarray):
                assert len(inputs[properties.n_atoms]) == len(
                    walk
                ), "Input data must contain as many collated frames as input walks"
                offset = 0
                for i, w in enumerate(walk):
                    mask[w[0] + offset] = 1
                    offset += inputs[properties.n_atoms][i]
            # single-walk interpretation
            else:
                mask[walk[0]] = 1
            q = q * mask + (1 - mask) * q.data

        for layer_idx, (interaction, mixing) in enumerate(
            zip(
                self.model.representation.interactions, self.model.representation.mixing
            )
        ):
            q, mu = interaction(q, mu, filter_list[layer_idx], dir_ij, idx_i, idx_j, n_atoms)
            mu = mu.detach()
            q, mu = mixing(q, mu)

            if walk is not None:
                mask = torch.zeros(q.shape).to(self.device)
                # multi-walk interpretation
                if isinstance(walk[0], tuple) or isinstance(walk[0], list) or isinstance(walk[0], np.ndarray):
                    offset = 0
                    for i, w in enumerate(walk):
                        mask[w[layer_idx + 1] + offset] = 1
                        offset += inputs[properties.n_atoms][i]
                # single-walk interpretation
                else:
                    mask[walk[layer_idx + 1]] = 1
                q = q * mask + (1 - mask) * q.data

        q = q.squeeze(1)

        inputs["scalar_representation"] = q
        inputs["vector_representation"] = mu

        return inputs, h0

    def so3net_forward(
        self, inputs: Dict[str, torch.Tensor], walk: Optional[List] = None
    ):
        # get tensors from input dictionary
        atomic_numbers = inputs[properties.Z]
        r_ij = inputs[properties.Rij]
        idx_i = inputs[properties.idx_i]
        idx_j = inputs[properties.idx_j]

        # compute atom and pair features
        d_ij = torch.norm(r_ij, dim=1, keepdim=True)
        dir_ij = r_ij / d_ij

        Yij = self.model.representation.sphharm(dir_ij)
        radial_ij = self.model.representation.radial_basis(d_ij)
        cutoff_ij = self.model.representation.cutoff_fn(d_ij)[..., None]

        x0 = self.model.representation.embedding(atomic_numbers)[:, None]

        # store type embedding
        x0 = Variable(x0.data, requires_grad=True)
        h0 = x0

        # do masking
        if walk is not None:
            mask = torch.zeros(x0.shape).to(self.device)
            # multi-walk interpretation
            if isinstance(walk[0], tuple) or isinstance(walk[0], list) or isinstance(walk[0], np.ndarray):
                assert len(inputs[properties.n_atoms]) == len(
                    walk
                ), "Input data must contain as many collated frames as input walks"
                offset = 0
                for i, w in enumerate(walk):
                    mask[w[0] + offset] = 1
                    offset += inputs[properties.n_atoms][i]
            # single-walk interpretation
            else:
                mask[walk[0]] = 1
            x0 = x0 * mask + (1 - mask) * x0.data

        x = so3.scalar2rsh(x0, int(self.model.representation.lmax))

        for so3conv, mixing1, mixing2, gating, mixing3 in zip(
            self.model.representation.so3convs,
            self.model.representation.mixings1,
            self.model.representation.mixings2,
            self.model.representation.gatings,
            self.model.representation.mixings3,
        ):
            dx = so3conv(x, radial_ij, Yij, cutoff_ij, idx_i, idx_j)
            ddx = mixing1(dx)
            dx = dx + self.model.representation.so3product(dx, ddx)
            dx = mixing2(dx)
            dx = gating(dx)
            dx = mixing3(dx)
            x = x + dx

            if walk is not None:
                mask = torch.zeros(q.shape).to(self.device)
                # multi-walk interpretation
                if isinstance(walk[0], tuple) or isinstance(walk[0], list) or isinstance(walk[0], np.ndarray):
                    offset = 0
                    for i, w in enumerate(walk):
                        mask[w[layer_idx + 1] + offset] = 1
                        offset += inputs[properties.n_atoms][i]
                # single-walk interpretation
                else:
                    mask[walk[layer_idx + 1]] = 1
                q = q * mask + (1 - mask) * q.data

        inputs["scalar_representation"] = x[:, 0]
        inputs["multipole_representation"] = x

        # extract cartesian vector from multipoles: [y, z, x] -> [x, y, z]
        if self.model.representation.return_vector_representation:
            inputs["vector_representation"] = torch.roll(x[:, 1:4], 1, 1)

        return inputs, h0

    def _zero_bias(self):
        for name, param in self.model.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)


class ProcessRelevancePope(ProcessRelevance):
    def __init__(self, model, device, target, gamma, zero_bias=False):
        super(ProcessRelevancePope, self).__init__(
            model, device, target, gamma, zero_bias=zero_bias
        )

    def process(self, sample):
        inputs = sample

        # reset grad
        self.model.zero_grad()

        for m in self.model.input_modules:
            inputs = m(sample)
        inputs, h0 = self.representation_forward(inputs)
        for m in self.model.output_modules:
            inputs = m(inputs)
        inputs = self.model.postprocess(inputs)
        y = inputs[self.target]
        # backward pass
        y.backward(retain_graph=True)

        # get node relevance
        node_relevances = (h0.data * h0.grad).sum(-1)
        # convert node relevance to self loop relevance
        all_relevances = []
        for node_idx, node_rel in enumerate(node_relevances):
            self_loop = [node_idx, node_idx]
            all_relevances.append((self_loop, node_rel.item()))

        # reset grad
        h0.grad.data.zero_()

        return all_relevances, y


class ProcessRelevanceGNNLRP(ProcessRelevance):
    def __init__(self, model, device, target, gamma, zero_bias=False):
        super(ProcessRelevanceGNNLRP, self).__init__(
            model, device, target, gamma, zero_bias=zero_bias
        )

    def process(self, sample, all_walks=None, batchsize=1):
        adj = torch.zeros((len(sample[properties.Z]), len(sample[properties.Z])))
        for idx_i, idx_j in zip(sample[properties.idx_i], sample[properties.idx_j]):
            adj[idx_i, idx_j] = 1
            adj[idx_j, idx_i] = 1
        # add diagonal
        adj += torch.eye(len(sample[properties.Z]))
        adj = adj.to(self.device) if "cuda" in self.device else None
        sample["adjacency"] = adj

        if all_walks is None:
            # No pre-information given, so let's compute all possible walks.
            all_walks = get_all_walks(
                len(self.model.representation.interactions) + 1, adj, self_loops=True
            )

        # collate inputs to match batchsize
        if batchsize > 1:
            # workaround since _atoms_collate_fn only works on cpu tensors
            input_devices = {}
            for k, v in sample.items():
                input_devices[k] = sample[k].device
                sample[k] = v.to("cpu")
            inputs = _atoms_collate_fn([sample for _ in range(batchsize)])
            for k, v in inputs.items():
                inputs[k] = v.to(input_devices[k])
        else:
            inputs = sample

        n_atoms_per_walk = len(all_walks[0])
        all_relevances = np.zeros((len(all_walks), n_atoms_per_walk+1))
        # divide walks into batches
        walk_batches = list(chunker(all_walks, batchsize))
        walk_idx = 0
        for walks in tqdm(walk_batches, mininterval=100):
            # the last batch might be smaller than the rest
            # thus we need to collate the data again
            if len(walks) < batchsize:
                # workaround since _atoms_collate_fn only works on cpu tensors
                inputs = _atoms_collate_fn([sample for _ in range(len(walks))])
                for k, v in inputs.items():
                    inputs[k] = v.to(input_devices[k])

            # if the batch contains a single walk continue with single-walk interpretation
            single_walk = False
            if len(walks) == 1:
                single_walk = True
                walks = walks[0]

            # reset grad
            self.model.zero_grad()

            for m in self.model.input_modules:
                inputs = m(inputs)
            inputs, h0 = self.representation_forward(inputs, walks)
            for m in self.model.output_modules:
                inputs = m(inputs)
            inputs = self.model.postprocess(inputs)

            y = inputs[self.target]

            # backward pass
            torch.sum(y).backward(retain_graph=True)

            # store walks relevance
            relevances = h0.data.cpu() * h0.grad.data.cpu()
            if single_walk:
                relevance = relevances.sum().item()
                all_relevances[walk_idx, :n_atoms_per_walk] = walks
                all_relevances[walk_idx, -1] = relevance
                walk_idx += 1
                # all_relevances.append((walks, relevance))
            else:
                n_0, n_1 = 0, inputs[properties.n_atoms][0].item()
                for i, w in enumerate(walks):
                    relevance = relevances[n_0:n_1].sum().item()
                    all_relevances[walk_idx, :n_atoms_per_walk] = w
                    all_relevances[walk_idx, -1] = relevance
                    walk_idx += 1
                    # all_relevances.append((w, relevance))
                    n_0 += inputs[properties.n_atoms][i]
                    if i != len(walks) - 1:
                        n_1 += inputs[properties.n_atoms][i + 1].item()

            # reset grad
            h0.grad.data.zero_()
        return all_relevances, y[-1]


###############################################################################
# postprocessing
###############################################################################
def take_most_relevant_walks(relevances, n_ats_per_walk=None, n_walks=50):
    # only consider n_atoms-walks
    if n_ats_per_walk is not None:
        relevances_tmp = []
        for walk_id, (walk, relevance) in enumerate(relevances):
            if len(set(walk)) == n_ats_per_walk:
                relevances_tmp.append((walk, relevance))
    else:
        relevances_tmp = relevances

    # sort relevances in descending order and take top n walks
    rels = [rel for _, rel in relevances_tmp]
    ids = np.argsort(-abs(np.array(rels))).tolist()
    most_relevant = []
    for i in ids[:n_walks]:
        most_relevant.append(relevances_tmp[i])
    return most_relevant


def scale_by_max(relevances):
    """
    Just scale the relevance values by the maximum.
    """
    max_abs_rel = max([abs(rel) for _, rel in relevances])
    # scale by max:
    relevances = [(walk, rel / max_abs_rel) for walk, rel in relevances]
    return relevances


def get_hop_relevances(all_relevances):
    # internal functions ###############################
    def _split_walk(walk):
        hops = []
        prev_node = None
        for node in walk:
            if prev_node is not None:
                hops.append([prev_node, node])
            prev_node = node
        return hops

    ###################################################

    relevances_tmp = []
    for walk, relevance in all_relevances:
        # determine scalar relevance value
        # split up walk into pairs of atoms (hops)
        hops = _split_walk(walk)
        for hop in hops:
            hop.sort()
            relevances_tmp.append((hop, relevance))

    return relevances_tmp


def aggregate_relevances(hop_relevances, absolute=False):
    relevances_tmp = []
    considered_hops = []
    for hop_ref, _ in hop_relevances:
        if hop_ref not in considered_hops:
            considered_hops.append(hop_ref)
            rel_tmp = 0
            for hop, relevance in hop_relevances:
                if hop_ref == hop:
                    if absolute:
                        rel_tmp += abs(relevance)
                    else:
                        rel_tmp += relevance
            relevances_tmp.append((hop_ref, rel_tmp))
    return relevances_tmp


# function to obtain relevances stats w.r.t. bond order
def get_bond_relevance(all_relevances, obmol, rdkmol):
    # list of atom objects in molecule
    _atom_list = []
    _aromatic_list = []
    for obatom in ob.OBMolAtomIter(obmol):
        _atom_list.append(obatom)
    for rdkatom in rdkmol.GetAtoms():
        _aromatic_list.append(rdkatom.GetIsAromatic())
    # 1. step: define list of tuples (hop, walk_relevance)
    relevances_tmp = get_hop_relevances(all_relevances)
    # 2. step: define list of tuples (bond_order, walk_relevance)
    relevances_tmp = get_bond_order_relevance(
        relevances_tmp, _atom_list, _aromatic_list
    )
    return relevances_tmp


# replace hop by bond order in <relevances>
def get_bond_order_relevance(relevances, atom_list, aromatic_list):
    relevances_tmp = []
    for hop, relevance in relevances:
        at_i = atom_list[hop[0]]
        at_j = atom_list[hop[1]]
        bond = at_i.GetBond(at_j)
        if bond is None:
            if hop[0] == hop[1]:
                bond_order = -1  # code for self loop
            else:
                bond_order = 0  # code for no bond
        elif aromatic_list[hop[0]] and aromatic_list[hop[0]]:
            bond_order = 1.5  # code for aromatic
        else:
            bond_order = bond.GetBondOrder()
        relevances_tmp.append((bond_order, relevance))
    return relevances_tmp


# function for filtering out walks that pass more or less than the specified number of atoms
def filter_by_walk_length(relevances, n_ats_per_walk):
    # only consider n-atoms walks
    if n_ats_per_walk is not None:
        relevances_tmp = []
        for w, r in relevances:
            if len(set(w)) == n_ats_per_walk:
                relevances_tmp.append((w, r))
        return relevances_tmp
    else:
        return relevances


# wrapper function for various above defined post-processing steps
def select_and_perform_post_processing(relevances, visualize_relevance, n_ats_per_walk):
    # take most relevant, aggregate or take all
    if visualize_relevance == "take_all":
        pass
    elif visualize_relevance == "aggregate_absolute":
        hop_relevances = get_hop_relevances(relevances)
        relevances = aggregate_relevances(hop_relevances, absolute=True)
    elif visualize_relevance == "aggregate":
        hop_relevances = get_hop_relevances(relevances)
        relevances = aggregate_relevances(hop_relevances, absolute=False)
    elif visualize_relevance == "most_relevant":
        relevances = take_most_relevant_walks(relevances, n_ats_per_walk)
    elif visualize_relevance == "max_scaling":
        relevances = scale_by_max(relevances)
    else:
        raise NotImplementedError("visualization mode not implemented")
    return relevances
