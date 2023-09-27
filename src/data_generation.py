import pandas as pd
#pd.set_option("display.max_columns", None)
import numpy as np
#import warnings
#from pandas.core.common import SettingWithCopyWarning
#warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
#warnings.simplefilter(action="ignore", category=FutureWarning)

#from time import time

#import torch

#import torchvision
#import torchvision.transforms as transforms

#import torch.nn as nn
#import torch.nn.functional as F
#import torch.optim as optim

#from torchvision import models

#import pyDOE

#from dexpy.samplers import uniform_simplex_sample
from scipy import spatial
#import matplotlib.pyplot as plt
import numpy as np





def sample_cond_unif(n_classes, target_sum):
    """Sample randomly a subset with a targeted overall sum of images."""
    proportions = np.random.sample(n_classes)
    subsets = np.round(target_sum * proportions / proportions.sum())
    return subsets


def determine_sample_distr(n_classes, n_max, class_max_counts):
    """Correct a given subset for the maximum number of images in a class."""
    subsets = sample_cond_unif(n_classes, n_max)
    while (subsets > class_max_counts).sum() > 0:
        argmax = np.argmax(subsets)
        overhead = subsets[argmax] - class_max_counts
        arg_smaller_class_max_counts = np.where(subsets < class_max_counts)
        tmp = sample_cond_unif(arg_smaller_class_max_counts[0].shape, overhead)
        subsets[arg_smaller_class_max_counts[0]] = subsets[arg_smaller_class_max_counts[0]] + tmp
        subsets[argmax] = subsets[argmax] - tmp.sum()
        #print(subsets)
    return subsets


def uniform_simplex_sample(N, q, clip=None):
    """Returns an array of points sampled uniformly from a simplex. 
    If needed, the initial exponential rng is clipped in order to have
    a higher likelihood to fullfill a constraint.

    :param N: the number of random sample to be generated
    :param q: the dimension of the simplex
    """
    sample = np.random.exponential(1.0, (N, q))
    if clip is not None:
        sample = np.clip(sample, 0, clip)
    row_sums = sample.sum(axis=1)
    sample = sample / row_sums[:, np.newaxis]
    return sample

def create_constrained_mixture_design(d, n, n_sum, c_max, n_optim, n_batch_size):
    """Create a constrained mixture design, optimized pointwise wrt the maximin criterion."""
    n_sample = 0
    # create a first doe
    clip = None
    loop_no = 0
    while n_sample < n:
        candidate = uniform_simplex_sample(n_batch_size, d, clip=clip) 
        candidate_scaled = candidate * n_sum
        # kick out non-acceptable rows:
        matches = np.where((candidate_scaled <= c_max).sum(axis=1) == d)[0]
        if matches.shape[0] > 0:
            candidate = candidate[matches,:]
            candidate_scaled = candidate_scaled[matches,:]
            if n_sample == 0:
                if candidate.shape[0] <= n:
                    doe_best = candidate
                    doe_best_scaled = candidate_scaled
                else:
                    doe_best = candidate[:n,:]
                    doe_best_scaled = candidate_scaled[:n,:]
                n_sample = doe_best.shape[0]
            else:
                doe_best = np.append(doe_best, candidate, axis = 0)
                doe_best_scaled = np.append(doe_best_scaled, candidate_scaled, axis=0)
                n_sample = doe_best.shape[0]
        else:
            if loop_no == 0:
                clip = 10
                loop_no += 1
            clip = clip * 0.9
            print(f"clip = {clip}")

    # reduce if oversampled:
    if n_sample > n:
        doe_best = doe_best[:n, :]
        doe_best_scaled = doe_best_scaled[:n, :]
    Mm = 0
    # improve the doe iteratively:
    i_better = 0
    for i in range(n_optim):
        candidate = doe_best
        candidate_scaled = doe_best_scaled
        dm = spatial.distance_matrix(candidate, candidate)
        np.fill_diagonal(dm, 20)
        dm_argmin = dm.min(axis=1).argmin()
        n_sample = 0
        while n_sample == 0:
            candidate_rows = uniform_simplex_sample(n_batch_size, d, clip=clip) 
            candidate_rows_scaled = candidate_rows * n_sum
            # kick out non-acceptable rows:
            matches = np.where((candidate_rows_scaled <= c_max).sum(axis=1) == d)[0]
            if matches.shape[0] > 0:
                n_sample = 1
                # in this point wise exchange algo we only need one point:
                candidate_row = candidate_rows[matches[0]]
                candidate_row_scaled = candidate_rows_scaled[matches[0]]
            else:
                if loop_no == 0:
                    clip = 10
                    loop_no += 1
                clip = clip * 0.95
                print(f"clip = {clip}")
        candidate[dm_argmin,:] = candidate_row
        candidate_scaled[dm_argmin,:] = candidate_row_scaled
        if dm.min() > Mm:
            i_better += 1
            if i_better % 100 == 0:
                print(dm.min(), i_better, i)
            doe_best = candidate
            doe_best_scaled = candidate_scaled
            dm_best = dm
            Mm = dm.min()
    return np.round(doe_best_scaled)

