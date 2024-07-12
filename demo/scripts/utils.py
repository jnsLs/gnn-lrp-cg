import numpy as np 
import pandas as pd

def compute_mean_std(arr,n_bins=75):
    '''
        Function call to bin values in column 0 based on values in column 1
        Inputs
            arr 
                column 0 -- values to be averaged inside bins
                column 1 -- values to be binned
        Returns
            mean
                average values inside each bin
            std
                standard error; sigma/sqrt(N)
            bin
    '''
    relev = arr[:,0]
    dists = arr[:,1]
    bins = np.linspace(dists.min(),dists.max()+1e-4,num=n_bins)
    labels=np.digitize(dists,bins)
    df = pd.DataFrame(dict(labels=labels,values=relev))
    mean = df.groupby('labels').mean().values
    std = df.groupby('labels').std().values / np.sqrt(len(df.groupby('labels').std().values))
    mask = df.groupby('labels').mean().index - 1
    bins = bins[mask]
    return mean[1:], std[1:], bins[1:]