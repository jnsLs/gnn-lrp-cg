import torch
import numpy as np
import matplotlib.pyplot as plt
from ase import Atoms
from rdkit import Chem
from rdkit.Chem import AllChem
from ase.io import read

import schnetpack as spk
from schnetpack.utils import load_model
from gnn_lrp_qc.interpretation.process_relevance import (
    #ProcessRelevancePope,
    ProcessRelevanceGNNLRP,
    #filter_by_walk_length,
    select_and_perform_post_processing,
)
from gnn_lrp_qc.utils.molecular_graph import Molecule
from gnn_lrp_qc.utils.visualization import relevance_vis_2d
import matplotlib
matplotlib.use("TkAgg")

# set up relevance calculation parameters
device = "cpu"
target_property = "energy_U0"
modelpath = "/home/jonas/Documents/xai/saved_models/qm9_new_schnet_2_5/best_inference_model"    # schnet
#modelpath = "/home/jonas/Documents/6-xai-cg/training/runs/8ef4a808-da1f-11ee-ab06-a86daa816ce9/best_model"  # painn
#modelpath = "/home/jonas/Documents/6-xai-cg/training/runs/0ae983dc-da36-11ee-af00-a86daa816ce9/best_model"  # so3net
smiles = "CC(=O)NC1=CC=C(C=C1)O"    # paracetamol
#smiles = "CCO"   # ethanol
RelevanceProcessor = ProcessRelevanceGNNLRP
zero_bias = False
gamma = 0.1
use_bias_rule_and_gamma = False

# Set up parameters visualization
aggregate = False

# load model
model = load_model(modelpath, device=device)
#model = torch.load(modelpath, map_location=torch.device(device))
model.do_postprocessing = False
cutoff = model.representation.cutoff.item()

# load data
rdkmol = Chem.MolFromSmiles(smiles)
rdkmol = Chem.AddHs(rdkmol)
AllChem.EmbedMolecule(rdkmol, randomSeed=42)
AllChem.UFFOptimizeMolecule(rdkmol)
# AllChem.MMFFOptimizeMolecule(rdkmol)
# get positions
n_nodes = rdkmol.GetNumAtoms()
pos = []
for i in range(n_nodes):
    conformer_pos = rdkmol.GetConformer().GetAtomPosition(i)
    pos.append([conformer_pos.x, conformer_pos.y, conformer_pos.z])
# get atomic numbers
atomic_numbers = []
for atom in rdkmol.GetAtoms():
    atomic_numbers.append(atom.GetAtomicNum())

at = Atoms(positions=np.array(pos), numbers=np.array(atomic_numbers))

atoms_converter = spk.interfaces.AtomsConverter(
    neighbor_list=spk.transform.MatScipyNeighborList(cutoff=cutoff),
    device=device,
)
#at = read("/home/jonas/Documents/4-momonano/benchmark_training/data/qm9.db")

sample = atoms_converter(at)


# process relevances
pr = RelevanceProcessor(
    model, device, target_property, gamma=gamma, use_bias_rule_and_gamma=use_bias_rule_and_gamma, zero_bias=zero_bias
)
relevances, y = pr.process(sample, batchsize=10)#, all_walks=all_walks)

r_tot = 0.0
for r in relevances:
    r_tot += r[-1]

#relevances_tmp = []
#for idx0_ref in range(20):
#    r_tmp = 0
#    for r in relevances:
#        idx0 = r[0]
#        if idx0 == idx0_ref:
#            r_tmp += r[-1]
#    relevances_tmp.append([float(idx0_ref), float(idx0_ref), -r_tmp])
#relevances = np.array(relevances_tmp)

relevances_plotting_format = []
for r in relevances:
    relevances_plotting_format.append(([int(r_) for r_ in r[:-1]], r[-1]))

print("total relevance: ", r_tot, "\n", "model output: ", y.item())

relevances = relevances_plotting_format

if aggregate:
    relevances = select_and_perform_post_processing(relevances, "aggregate", None)

# VISUALIZE
# embed molecule structure
molecule = Molecule(sample["_positions"], sample["_atomic_numbers"])
rdkmol = molecule.rdkmol
graph = molecule.con_mat
pos_2d = molecule.embed_in_2d()

# embed sample["_positions"] in 2d
#pos_2d = sample["_positions"][:, :2].detach().cpu().numpy()
#graph = None

# initialize figure
fig = plt.figure(figsize=(14, 8))
ax = plt.subplot(1, 1, 1)

# plot
ax = relevance_vis_2d(
    ax,
    relevances,
    sample["_atomic_numbers"],
    pos_2d,
    graph,
    cmap=None,
    relevance_scaling=0.2,
    scaling_type="root",
    shrinking_factor=1,
)
plt.axis("off")
plt.savefig("/tmp/paracetamol_relevance.pdf")
plt.show()

