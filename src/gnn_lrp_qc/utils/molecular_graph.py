import torch
import numpy as np
from rdkit.Chem import AllChem
from gnn_lrp_qc.utils.xyz2mol import convert


class Molecule:
    """
    pos: (dim: n_at x 3)
    at_num: (dim: n_at)
    """

    def __init__(self, pos, at_num):
        self.rdkmol, self.con_mat = convert(
            atoms=at_num.tolist(), xyz_coordinates=pos.tolist(), charge=0
        )
        self.n_nodes = self.rdkmol.GetNumAtoms()

    def embed_in_2d(self):
        # compute 2D embedding
        AllChem.Compute2DCoords(self.rdkmol)
        # compute 2D positions
        pos = []
        for i in range(self.n_nodes):
            conformer_pos = self.rdkmol.GetConformer().GetAtomPosition(i)
            pos.append([conformer_pos.x, conformer_pos.y])
        pos = np.array(pos)
        return pos


class Tree(object):
    """
    Reused tree object from stanfordnlp/treelstm.
    """

    def __init__(self):
        self.parent = None
        self.num_children = 0
        self.children = list()

    def add_child(self, child):
        child.parent = self
        self.num_children += 1
        self.children.append(child)

    def size(self):
        if getattr(self, "_size"):
            return self._size
        count = 1
        for i in xrange(self.num_children):
            count += self.children[i].size()
        self._size = count
        return self._size

    def depth(self):
        if getattr(self, "_depth"):
            return self._depth
        count = 0
        if self.num_children > 0:
            for i in xrange(self.num_children):
                child_depth = self.children[i].depth()
                if child_depth > count:
                    count = child_depth
            count += 1
        self._depth = count
        return self._depth

    def __iter__(self):
        yield self
        for c in self.children:
            for x in c:
                yield x


def get_all_walks(L, lamb, end_id=None, node_id=None, self_loops=True):
    """
    :param node_id: The id of the node we start with.
    :param L: The length of every walk.
    :param lamb: The adjacency matrix of the graph we consider.
    """
    if L == 1:
        return [[idx, idx] for idx in range(lamb.shape[-1])]

    def get_seq_of_nodes(tree):
        node_seq = [tree.id]
        while tree.parent is not None:
            tree = tree.parent
            node_seq.append(tree.id)
        return node_seq[::-1]

    def get_neighbors(id, lamb):
        x = torch.zeros(lamb.shape[1], 1)
        x[id, 0] = 1.0
        neighbors = lamb.mm(x).nonzero()[:, 0]
        return [int(id) for id in neighbors]

    if node_id is None:
        # Multiple start nodes
        num_of_nodes = lamb.shape[1]
        current_nodes = [None] * num_of_nodes
        for i in range(num_of_nodes):
            current_nodes[i] = Tree()
            current_nodes[i].id = i
    else:
        # Starting in one node
        root = Tree()
        root.id = node_id
        current_nodes = [root]

    for l in range(L - 1):
        leaf_nodes = []
        for node in current_nodes:
            for neighbor in get_neighbors(node.id, lamb):
                new_node = Tree()
                new_node.id = neighbor
                node.add_child(new_node)
                leaf_nodes.append(new_node)

        current_nodes = leaf_nodes

    all_walks = []
    for node in leaf_nodes:
        if end_id is None:
            all_walks.append(get_seq_of_nodes(node))
        else:
            if node.id == end_id:
                all_walks.append(get_seq_of_nodes(node))

    # filter out walks that include self loops
    if not self_loops:
        all_walks_filtered = []
        for w in all_walks:
            if len(set(w)) == len(w):
                all_walks_filtered.append(w)
        return all_walks_filtered

    return all_walks


class PairwiseDistances(torch.nn.Module):
    """
    Layer for computing the pairwise distance matrix. Matrix is quadratic
    and symmetric with entries d_ij = r_i - r_j. All possible atom pairs are
    considered. This Layer is particularly used in the clustering module.

    Returns:
        torch.Tensor: Pairwise distances (Nbatch x Nat x Nat)

    """

    def __init__(self):
        super(PairwiseDistances, self).__init__()

    def forward(self, positions):
        n_nodes = positions.shape[1]
        # compute tensor distance tensor with entries d_ij = r_i - r_j
        positions_i = positions.unsqueeze(2).repeat(1, 1, n_nodes, 1)
        positions_j = positions.repeat(1, n_nodes, 1).view(-1, n_nodes, n_nodes, 3)
        dist_vecs = positions_i - positions_j
        r_ij = dist_vecs.norm(dim=-1, p=2)
        return r_ij, dist_vecs


def molecule_distortion(pos_src, stretch_coeff, atom_pair):
    # get binary graph
    distances = PairwiseDistances()
    r_ij, dist_vecs = distances(pos_src)
    graph = (r_ij < 1.6).float()[0].cpu().numpy()
    graph -= np.eye(graph.shape[0])

    n_atoms = pos_src.shape[1]
    idx0 = atom_pair[0]
    idx1 = atom_pair[1]

    # find atoms affected by stretching
    # dot = torch.matmul(dist_vecs[0, 0], dist_vecs[0, 0].transpose(0, 1))
    # dot = torch.matmul(dist_vecs[0, idx0], dist_vecs[0, idx0].transpose(0, 1))
    # affected_atoms = dot[8] > 0.

    affected_atoms = torch.tensor([False] * n_atoms, device="cuda:0")
    affected_atoms[idx1] = True

    pos_post = pos_src[0]
    # shift affected atoms
    for idx, is_affected in enumerate(affected_atoms):
        if is_affected:
            pos_post[idx] = (
                pos_post[idx]
                + dist_vecs[0, idx1, idx0] / r_ij[0, idx1, idx0] * stretch_coeff
            )

    return graph, pos_post
