This folder contains a small demo code for running an LRP interpretation on a pre-trained model.

The model is provided as a schnetpack checkpoint in the `model` folder. As a test here we use the so3net model for methane shown in the manuscript, that is fastest to interpret due to its small cutoff and the relatively small number of CG beads in the system. 

On our system, running on Debian GNU/Linux 12 (bookworm) with Intel(R) Core(TM) i9-10900 CPU @ 2.80GHz 10 cores and a single NVIDIA GeForce RTX 3090 GPU with CUDA 12.5, the interpretation takes about 25 minutes. The pre-and post-processing steps take only a few seconds to run.

Note that if you are running the code on a computer without GPU, the same installation instructions remain but the interpretation is expected to take about 90 hours on a 2020 MacBook Pro with 2.3 GHz Intel Core i7 four cores. We do not advise running the demo without a GPU.

## Installation instructions

Before running the demo, please install the following software on a computer with GPU in a fresh python environment:

```
conda create -n cg-lrp-test python=3.11 -y
```

Install SchNetPack: 
```
git clone https://github.com/atomistic-machine-learning/schnetpack.git 
cd schnetpack
pip install .
```

Clone the present gnn-lrp-cg repo and install the package in your environment:
```
cd gnn-lrp-cg
pip install .
```

Install a few additional packages required for plotting:
```
conda install -c conda-forge pandas jupyter ipython
python -m ipykernel install --user --name cg-lrp-test --display-name "cg-lrp-test"
```

## Running instructions

This demo shows how to run the interpretation of a single methane frame with the so3net model shown in the manuscript. The methane frame is present as a pickle file (containing a dictionary in the format readable by SchNetPack) in `demo/interpretation/frames/frame_0.pkl`.

The scripts to run the different steps of the interpretation are provided in `demo/scripts` and are numbered in the order in which they should be run.

First, precompute the walks in the frame to be interpreted:
```
cd scripts
conda activate cg-lrp-test

python 0_get_walks_per_frame.py
```
This scripts computes the walks associated with each numbered frame in `demo/interpretation/frames` and saves them in `demo/interpretation/walks_per_frame`. Note that this script has a commented part that allows to cut the walks into different chunks to be run in parallel. For this demo, this part should not be necessary, but should you still opt for uncommenting this part, make sure to change the value of `n_GPUs_per_frame` to 4 in the next script `1_run_interpretation.py` (line 15).

Once the walks are pre-computed, run LRP on each walk:
```
python 1_run_interpretation.py 0
```
The argument `0` computes the interpretation for chunk 0 of frame 0. Should you want to interpret more frames and/or cut the walks in batches, this argument can be changed for parallel runs on different GPUs. For the manuscript, jobs were run on slurm, submitting N jobs as a job array with the argument varying from 0 to N-1.

Note that, depending on the amount of available memory on your GPU, the batchsize in the relevance computation (line 68) might have to be adjusted. For reference, the batch size of 10 set in the script was set for running on an NVIDIA RTX 3090 with 24GB memory.

This script saves a file in `demo/interpretation/interpretations_per_frame` for each frame (and chunk if set) containing the raw relevance output. This is saved in a `.npy` file containing a dictionary with keys:

* `relevance` containing an array of shape (n_walks, len(walk)+1) containing for each walk the bead indices of the walk and the corresponding relevance score, 
* `energy` containing the model output for the interpreted frame.

Once the interpretation has been run, one has access to the relevance score for each walk. In order to facilitate the plotting of the results, a post-processing is done where the relevance values are normalized, sorted into 2- and 3-body walks and relevant features (such as distances or angles) are computed.
This is done through:
```
python 2_process_interpretation.py
```
This script creates a dictionary with keys `2` and `3` for the 2- and 3-body walks respectively. Each key contains a numpy array with properties for each walk, see the docstring in `2_process_interpretation.py` for more details.

The expected result is shown in `demo/interpretation/expected_results.png` and the same plot can be reproduced on your interpretation using `3_plot_interpretation.ipynb`.


## To reproduce the results in the manuscript

Follow the reproduction guidelines provided with the supplementary data. The data is organized in a similar way to this demo and the corresponding scripts to reproduce the results are provided in each `interpretation` folder for each system/model.
