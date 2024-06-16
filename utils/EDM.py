import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from matplotlib.colors import LinearSegmentedColormap
import torch


alpha = 1.2  # Adjust as needed
beta = 4.0   # Adjust as needed

def generate_edm(image, alpha=2.0, beta=3.0):
    # Convert the image to binary (0 for background, 1 for foreground)
    
    binary_image = np.where(image.cpu().numpy() > 0, 1, 0)

    # Compute the distance transform for foreground and background
    distance_transform_fg = scipy.ndimage.distance_transform_edt(binary_image)  #边界是0   中心是大的
    distance_transform_bg = scipy.ndimage.distance_transform_edt(1 - binary_image)

    # print(distance_transform_fg.max(),distance_transform_fg.min())

    weight_map_fg = alpha * np.exp(-distance_transform_fg ** 2 / beta ** 2)
    weight_map_fg[binary_image==0] = 0

    weight_map_bg = alpha * np.exp(-distance_transform_bg ** 2 / beta ** 2)
    weight_map_bg[binary_image==1] = 0

    weight_map = weight_map_fg + weight_map_bg
    # weight_map = weight_map + 0.1
    weight_map = torch.from_numpy(weight_map).float()
    
    return weight_map

