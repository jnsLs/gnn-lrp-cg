from itertools import combinations_with_replacement, combinations
import numpy as np
import torch
from tqdm import tqdm
import pickle
from gnn_lrp_qc.utils.geometry import compute_cell_shifts, compute_distances
from schnetpack import properties
from glob import glob


def create_relevance_array(interpretation_path_list, frames_path_list, rcut, normalize="total_rel"):
    """
    This function processes the raw LRP output to facilitate plotting. 
    It returns a dictionnary with 2-body relevances and filtered 3-body relevances depending on the provided cutoff for the first solvation shell

    Parameters:
    -----------
        interpretation_path_list: List[str]
            list of raw interpretation output files (.npy files)
        frames_path_list: List[str]
            list of .pkl files corresponding to interpreted frames. Will be used for computing distances. 
            Coordinates should be in angstrom (or make sure to change rcut and the labels on the plots accordingly)
        rcut: float
            value of the cutoff for the first solvation shell in angstrom (used to filter triplets of beads in the 3B walks)
        normalize: str or False
            option for normalizing. If False, no normalization is applied.
            "energy_abs" normalizes through the absolute value of the network output for each frame
            "total_rel" normalizes through the total relevance of each frame
            "total_rel_x_n_walks" normalizes through the absolute value of the total relevance for each frame divided by the total number of walks in that frame

    Returns: 
    --------
        relevance_distances: Dict
            dictionnary containing the processed relevances in the following format:
            keys 2 and 3 correspond to two- and three-body relevances respectively
            relevance_distances[2] is a numpy array of shape (n_2b_walks, 7) where for each line corresponds to a 2B walk and contains
                [relevance, r_{ij}, walk[0], walk[1], walk[2], walk[3], frame_id]
            relevance_distances[3] is a numpy array of shape (n_triplets, 10) where n_triplets is the number of triplets within the first solvation shell. Each line corresponds to a triplet/walk and contains
                [relevance, r_{ij}, r_{ik}, r_{jk}, theta_{ijk}, walk[0], walk[1], walk[2], walk[3], frame_id]
                for each 3B walk, this function looks at what permutation of the 3 nodes (ijk) is such that r_{ij} and r_{jk} are lower than rcut and computes theta_{ijk} accordingly. 


    """
    if "chunk" in interpretation_path_list[0]:
        assert np.all(["chunk" in fn for fn in interpretation_path_list]), "run interpretation for all frames or for chunks, not for both, otherwise values will be counted twice."
        frames = [fn.split("../interpretation/interpretation_frame_")[-1].split("_chunk")[0] for fn in interpretation_path_list]
    else:
        assert np.all(["all" in fn for fn in interpretation_path_list]), "run interpretation for all frames or for chunks, not for both, otherwise values will be counted twice."
        frames = [fn.split("../interpretation/interpretation_frame_")[-1].split("_all")[0] for fn in interpretation_path_list]

    frames = np.unique(frames)
    relevance_distances = {}

    # Do a first loop over all the files to store the total relevance per frame
    if normalize == "energy_abs":
        total_relevance = np.zeros(len(frames))
        for fn in tqdm(interpretation_path_list, desc=f"Computing total relevance per frame ..."):
            mol_data = np.load(fn, allow_pickle=True).item()

            if "chunk" in fn:
                frame_id = int(fn.split("/interpretation_frame_")[-1].split("_chunk")[0])
            else:
                frame_id = int(fn.split("/interpretation_frame_")[-1].split("_all")[0])
            output = mol_data["energy"]
            total_relevance[frame_id] = np.abs(output.item())
    elif normalize == "total_rel":
        total_relevance = np.zeros(len(frames))
        for fn in tqdm(interpretation_path_list, desc=f"Computing total relevance per frame ..."):
            mol_data = np.load(fn, allow_pickle=True).item()
            if "chunk" in fn:
                frame_id = int(fn.split("/interpretation_frame_")[-1].split("_chunk")[0])
            else:
                frame_id = int(fn.split("/interpretation_frame_")[-1].split("_all")[0])
            total_rel = np.sum(mol_data["relevances"], axis=0)[-1]
            total_relevance[frame_id] += total_rel
        total_relevance = np.abs(total_relevance)
    elif normalize == "total_rel_x_n_walks":
        total_relevance = np.zeros(len(frames))
        n_walks_per_frame = np.zeros(len(frames))
        for fn in tqdm(interpretation_path_list, desc=f"Computing total relevance per frame ..."):
            mol_data = np.load(fn, allow_pickle=True).item()
            if "chunk" in fn:
                frame_id = int(fn.split("/interpretation_frame_")[-1].split("_chunk")[0])
            else:
                frame_id = int(fn.split("/interpretation_frame_")[-1].split("_all")[0])
            total_rel = np.sum(mol_data["relevances"], axis=0)[-1]
            total_relevance[frame_id] += total_rel
            n_walks_per_frame[frame_id] += len(mol_data["relevances"])
        total_relevance = np.abs(np.divide(total_relevance, n_walks_per_frame))
        # print(total_relevance)
    elif normalize == False:
        total_relevance = np.ones(len(frames))
    
    # Do a loop over the frames to compute the distances
    frames_distances = {}
    for fn in tqdm(frames_path_list, desc="Computing distance matrices ..."):
        frame_id = int(fn.split("/frame_")[-1].split(".pkl")[0])
        with open(fn, "rb") as f:
            frame_data = pickle.load(f)
        
        atom_ids = torch.arange(frame_data[properties.position].shape[0])
        mapping = torch.cartesian_prod(atom_ids, atom_ids).t()
        mapping = mapping[:, mapping[0] != mapping[1]]
        pbc = frame_data[properties.pbc].reshape((1, *frame_data[properties.pbc].shape))
        cell_shifts = compute_cell_shifts(
            frame_data[properties.position], 
            mapping, 
            pbc, 
            frame_data[properties.cell], 
            frame_data[properties.idx_m]
        )
        distances = compute_distances(frame_data[properties.position],mapping,cell_shifts).cpu().detach().numpy()
        distance_matrix = np.zeros((frame_data[properties.position].shape[0],frame_data[properties.position].shape[0]))
        for dist,edge in zip(distances,mapping.T):
            distance_matrix[edge[0],edge[1]] = dist
            distance_matrix[edge[1],edge[0]] = dist
        frames_distances[frame_id] = distance_matrix
    frame_path_dict = {}
    for path in frames_path_list:
        frame_id = int(path.split("frame_")[-1].split(".pkl")[0])
        frame_path_dict[frame_id] = path

    
    ## process the relevance to gather the information for 2- and 3-body walks

    for n_atoms_per_walk in [2, 3]: #[2,3,4]
        relevance_distances[n_atoms_per_walk] = []
        for fn in tqdm(interpretation_path_list, desc=f"Processing {n_atoms_per_walk}-body walks..."):
            mol_data = np.load(fn, allow_pickle=True).item()

            if "chunk" in fn:
                frame_id = int(fn.split("/interpretation_frame_")[-1].split("_chunk")[0])
            else:
                frame_id = int(fn.split("/interpretation_frame_")[-1].split("_all")[0])

            frame_data_fn = frame_path_dict[frame_id]
            assert f"frame_{frame_id}.pkl" in frame_data_fn, "sort didnt work as expected"

            distance_matrix = frames_distances[frame_id]

            with open(frame_data_fn, "rb") as f:
                frame_data = pickle.load(f)
            
            frame_positions = frame_data[properties.position].detach().cpu().numpy()
            for line in mol_data['relevances']:
                walk = np.array(line[:-1], dtype=int)
                rel = line[-1]
                if len(set(walk)) == n_atoms_per_walk:
                    atoms = []
                    #append unique atoms in the right order
                    for idd in list(dict.fromkeys(walk)):
                        atoms.append(idd)
                    atom_pairs = []
                    dists = []

                    for id0, id1 in combinations(atoms, 2):
                        atom_pairs.append([id0, id1])
                        dists.append(distance_matrix[id0, id1])
                    if n_atoms_per_walk == 3:

                        #theta_ijk
                        rij, rik, rjk = dists

                        if rij < rcut and rik < rcut:
                            data_line = []
                            cos_theta = (rij**2 + rik**2 - rjk**2) / (2*rij*rik) #theta_{kij}
                            # Avoid discontinuities
                            cos_theta = np.clip(cos_theta,-0.9999,0.9999)
                            theta = np.arccos(cos_theta)*180/np.pi
                            data_line.extend(dists)
                            data_line.append(theta)
                            data_line.extend(walk)
                            data_line.append(frame_id)
                            relevance_distances[n_atoms_per_walk].append([rel/total_relevance[frame_id],*data_line])

                        if rij < rcut and rjk < rcut:
                            data_line = []
                            cos_theta = (rij**2 + rjk**2 - rik**2) / (2*rij*rjk) #theta_{ijk}
                            cos_theta = np.clip(cos_theta,-0.9999,0.9999)
                            theta = np.arccos(cos_theta)*180/np.pi
                            data_line.extend(dists)
                            data_line.append(theta)
                            data_line.extend(walk)
                            data_line.append(frame_id)
                            relevance_distances[n_atoms_per_walk].append([rel/total_relevance[frame_id],*data_line])

                        if rik < rcut and rjk < rcut:
                            data_line = []
                            cos_theta = (rik**2 + rjk**2 - rij**2) / (2*rik*rjk) #theta_{jki}
                            cos_theta = np.clip(cos_theta,-0.9999,0.9999)
                            theta = np.arccos(cos_theta)*180/np.pi
                            data_line.extend(dists)
                            data_line.append(theta)
                            data_line.extend(walk)
                            data_line.append(frame_id)
                            relevance_distances[n_atoms_per_walk].append([rel/total_relevance[frame_id],*data_line])

                    if n_atoms_per_walk == 2:
                        data_line = []
                        data_line.extend(dists)
                        data_line.extend(walk)
                        data_line.append(frame_id)
                        relevance_distances[n_atoms_per_walk].append([rel/total_relevance[frame_id],*data_line])
                    

        relevance_distances[n_atoms_per_walk] = np.array(relevance_distances[n_atoms_per_walk])
    return relevance_distances

if __name__ == "__main__":

    interpretation_path_list = sorted(glob("../interpretation/interpretations_per_frame/interpretation_frame*.npy"))
    frames_path_list = sorted(glob("../interpretation/frames/frame_*.pkl"))

    rcut = 0.5608108*10
    print(f"\nComputing relevances with rcut {rcut} AA\n")

    relevance_array = create_relevance_array(interpretation_path_list, frames_path_list, rcut, normalize="total_rel_x_n_walks")

    with open(f"../interpretation/relevance_array_so3net_2B_3B_methane_cut_{rcut:.4f}.pkl", "wb") as f:
        pickle.dump(relevance_array, f)
