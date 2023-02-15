from numba import jit
import numpy as np

side_len = 3
hist_bins = 3
volume = np.random.randint(0, 255, (side_len, side_len, side_len))

@jit(nopython=False) 
def histogram(kernels, bins, drop_zero_bin=True):
    hist = np.zeros((volume.shape[0], volume.shape[1], volume.shape[2], bins))

    for i in range(volume.shape[0]):
        for j in range(volume.shape[1]):
            for k in range(volume.shape[2]):
                hist[i, j, k, :] = np.histogram(kernels[i, j, k], bins=bins, range=(0, 255))[0]
    if drop_zero_bin:
        hist = hist[:, :, :, 1:]
    
    # Reshape to (bins, side_len, side_len, side_len)
    return np.squeeze(np.stack(np.split(hist, hist_bins-1, axis=3), axis=0))

hist = histogram(np.lib.stride_tricks.sliding_window_view(volume, (3, 3, 3)), hist_bins, drop_zero_bin=True)