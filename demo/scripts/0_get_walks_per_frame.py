import torch
import numpy as np 
import sys 
import pickle
import os
from glob import glob
from tqdm import tqdm
from schnetpack import properties
from gnn_lrp_qc.utils.molecular_graph import get_all_walks
from gnn_lrp_qc.utils.batch import chunker

# set up relevance calculation parameters
modelpath = "../model/best_model"

outdir = "../interpretation/walks_per_frame"
if not os.path.exists(outdir):
    os.mkdir(outdir)

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


# load model
model = torch.load(modelpath, map_location=torch.device(device))
model.do_postprocessing = False
cutoff = model.representation.cutoff.item()

#load data
if device == "cuda":
    fns = sorted(glob("../interpretation/frames/frame_*.pkl"))
else:
    fns = sorted(glob("../interpretation/frames/cpu_frame_*.pkl"))
for fn in tqdm(fns):
    frame_id = int(fn.split("/frame_")[-1].split(".pkl")[0])
    if not os.path.exists(os.path.join(outdir, f"frame_{frame_id}")):
        os.mkdir(os.path.join(outdir, f"frame_{frame_id}"))
    else:
        continue
    
    with open(fn, "rb") as f:
        sample = pickle.load(f)

    ## get the walks for which to compute relevances
    walks_for_rel = []

    #compute the graph adjacency matrix
    adj = torch.zeros((len(sample[properties.Z]), len(sample[properties.Z])))
    for idx_i, idx_j in zip(sample[properties.idx_i], sample[properties.idx_j]):
        adj[idx_i, idx_j] = 1
        adj[idx_j, idx_i] = 1
    # add diagonal
    adj += torch.eye(len(sample[properties.Z]))
    adj.to(device) if device == "cuda" else None

    # compute the walks depending on the number of layers
    all_walks = get_all_walks(len(model.representation.so3convs) + 1, adj, self_loops=True)
    np.save(os.path.join(outdir, f"frame_{frame_id}/frame_{frame_id}_all_walks.npy"), np.array(all_walks))
    
    ## uncomment this part if you want to split the walks into chunks to be run in parallel on different GPUs
    # if len(walks_for_rel)%4 == 0:
    #     n_walks_per_chunk = len(walks_for_rel)//4
    # else:
    #     n_walks_per_chunk = len(walks_for_rel)//4 + 1
    # chunks = list(chunker(walks_for_rel, n_walks_per_chunk))
    # for i, chunk in enumerate(chunks):
        # np.save(os.path.join(outdir, f"frame_{frame_id}/frame_{frame_id}_chunk_{i}.npy"), np.array(chunk))