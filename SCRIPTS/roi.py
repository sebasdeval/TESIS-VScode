"""
Utility functions for deep learning sound detection module
"""

import json
import logging
import os
import shutil
import torch
import numpy as np
import pandas as pd

#%% Functions for build_dataset.py

def roi2windowed2(wl, roi):
    """
    Split a single region of interest (roi) into multiple regions of fixed size according
    to a window length. If window length (wl) is longer than the roi, the result is a single
    roi of length wl and centered in the middle of the roi. If the window length is 
    shorter than the, the roi is splitted into multiple regions. Regions must have at
    least 50% of overlap with the new window length. There is no overlap between windows.
    
    Parameters
    ----------
    wl : float
        Window length of the resulting regions of interest
    roi : pandas.core.frame.DataFrame
        Regions of interest with at least five columns, min_t, max_t, min_f, max_f, label.

    Returns
    -------
    roi_fmt : pandas.core.frame.DataFrame
        Formated regions of interest with fixed size.

    """
    roi_len = (roi.max_t - roi.min_t)
    
    if roi_len < wl:
        # region shorter than window length
        roi_median = roi.min_t + roi_len/2
        roi.loc['min_t'] = roi_median - wl/2
        roi.loc['max_t'] = roi_median + wl/2
        roi_fmt = roi.to_frame().T
    
    else:
        # region larger than window length
        # compute arrays. If more than 50% overlap remains, add a window
        roi_fmt = pd.DataFrame({'min_t': np.arange(roi.min_t, roi.max_t-wl+(wl/2), wl),
                                 'max_t': np.arange(roi.min_t+wl, roi.max_t+(wl/2), wl),
                                 #'min_f': roi.min_f,
                                 #'max_f': roi.max_f,
                                 'label': roi.label})
    return roi_fmt
