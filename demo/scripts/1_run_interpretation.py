import torch
import numpy as np 
import sys 
import pickle
import os
from gnn_lrp_qc.interpretation.process_relevance import ProcessRelevanceGNNLRP
from schnetpack import properties
from gnn_lrp_qc.utils.molecular_graph import get_all_walks

#total numer of frames to compute the interpretation for
total_n_frames = 1

#number of GPUs to use for the computation of a single frame
#should be consistent with the output in walks_per_frame - change to 4 if you cut the walks into 4 chunks in get_walks_per_frame.py
n_GPUs_per_frame = 1

task_id = int(sys.argv[1])

frame_id, cuda_idx = np.unravel_index(task_id, (total_n_frames, n_GPUs_per_frame))

print(f"\nInterpreting frame {frame_id} chunk {cuda_idx}\n")

# set up relevance calculation parameters
modelpath = "../model/best_model"

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
target_property = "energy"

RelevanceProcessor = ProcessRelevanceGNNLRP
zero_bias = False

# load model
model = torch.load(modelpath, map_location=torch.device(device))
model.do_postprocessing = False
cutoff = model.representation.cutoff.item()

#load data
if device == "cuda":
    with open(f"../interpretation/frames/frame_{frame_id}.pkl", "rb") as f:
        sample = pickle.load(f)
    for k, v in sample.items():
        sample[k] = v.to(device)
        if k in ["_positions"]:
            sample[k] = v.requires_grad_(True)
else:
    with open(f"../interpretation/frames/cpu_frame_{frame_id}.pkl", "rb") as f:
        sample = pickle.load(f)
    for k, v in sample.items():
        sample[k] = v.to(device)
        if k in ["_positions"]:
            sample[k] = v.requires_grad_(True)

## get the walks for which to compute relevances
if n_GPUs_per_frame == 1:
    walks_for_rel = np.load(f"../interpretation/walks_per_frame/frame_{frame_id}/frame_{frame_id}_all_walks.npy").tolist()
else:
    walks_for_rel = np.load(f"../interpretation/walks_per_frame/frame_{frame_id}/frame_{frame_id}_chunk_{cuda_idx}.npy").tolist()

# create output directory
if not os.path.exists("../interpretation/interpretations_per_frame"):
    os.mkdir("../interpretation/interpretations_per_frame")

# process relevances
pr = RelevanceProcessor(model, device, target_property, gamma=0.1, use_bias_rule_and_gamma=True, zero_bias=zero_bias)
relevances, y = pr.process(sample, all_walks=walks_for_rel, batchsize=10)
mol_data = {"relevances": relevances, target_property: y.cpu().detach().numpy()}

if n_GPUs_per_frame == 1:
    np.save(f"../interpretation/interpretations_per_frame/interpretation_frame_{frame_id}_all_walks.npy", mol_data)
else:
    np.save(f"../interpretation/interpretations_per_frame/interpretation_frame_{frame_id}_chunk_{cuda_idx}.npy", mol_data)
