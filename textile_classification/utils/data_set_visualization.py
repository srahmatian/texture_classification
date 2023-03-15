
import matplotlib.pyplot as plt
from typing import Union
import torch
import math
from config import logger, LABELS_INT_To_STR
from textile_classification.utils import change_labels_from_probablity_format_to_deterministic_format

def plot_images(images: torch.Tensor, 
                labels_grnd_truth: torch.Tensor, 
                labels_predicted: Union[torch.Tensor, None]=None,
                threshold: float=0.5) -> plt.figure:
    """
    It takes a bunch of images and show them with their ground truth lable on the x-axis, 
    and predicted labelon the y-axis.
    If the prediction is correct, we show the prediction in green, otherwise in red.
    If you want to show images, and you do not have the predictions yet, 
    just left it unassigned.

    Args:
        images (torch.Tensor): a pytorch tensor in the shape of (batch_size, 3, height, width)
        labels_grnd_truth (torch.Tensor): a pytorch tensor in the shape of (batch_size, )
        labels_predicted (torch.Tensor): a pytorch tensor in the shape of (batch_size, )
                                         it doesn't matter if labels_predicted are in probability or deterministic forrmat. 

    Returns:
        plt.figure: It return the figure, so you can save or use it in another place.
    """
    # assert the input tensors are in the shape of having batch_size.
    assert (images.dim() == 4), logger.error("input images must be a tensor having 4 dimensions")
    assert (labels_grnd_truth.dim() == 1), logger.error("input labels_grnd_truth must be a tensor having 1 dimension")
    if labels_predicted is not None:
        assert (labels_predicted.dim() == 1), logger.error("input labels_predicted must be a tensor having 1 dimension")
    assert (images.shape[1] == 3), logger.error("input images must have three channels")
    
    batch_size = images.shape[0] # or labels_ground_truth.shape[0], or labels_predicted.shape[0]
    num_columns = 8
    num_rows = math.ceil(batch_size / num_columns)
    if (batch_size < 8):
        fig = plt.figure(figsize=(batch_size * 5, batch_size))
    else:    
        fig = plt.figure(figsize=(num_columns * 5, num_columns * num_rows))
    # note that the indices in fig.add_subplot must start from 1 not 0.
    for i in range(1, batch_size + 1):
        fig.add_subplot(num_rows, num_columns, i, xticks = [], yticks = [])
        # Remember we have transformed images from PIL (H, W, C) to Tensor (C, H, W), 
        # We need to convert back its shape to be able of showing it using matplotlib.
        each_image = images[i-1].numpy().transpose((1, 2, 0))
        plt.imshow(each_image)

        plt.xlabel(f"{LABELS_INT_To_STR[int(labels_grnd_truth[i-1])]}", fontsize=24)

        if labels_predicted is not None:
            labels_predicted = change_labels_from_probablity_format_to_deterministic_format(
                labels_predicted,
                is_one_hot_format=False, 
                threshold=threshold)
            color = "green"
            if labels_predicted[i-1] != labels_grnd_truth[i-1]:
                color = "red"
            plt.ylabel(f"{LABELS_INT_To_STR[int(labels_predicted[i-1])]}", color = color, fontsize=24)
    plt.tight_layout()
    return fig
