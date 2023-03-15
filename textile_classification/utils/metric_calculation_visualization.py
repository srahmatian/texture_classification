
from sklearn.metrics import (accuracy_score, precision_score, 
                             recall_score, f1_score, confusion_matrix, 
                             roc_curve, auc,
                             precision_recall_curve)
import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import Dict, List
from config import LABELS_INT_To_STR, logger

def change_labels_from_one_hot_fromat_to_binary_format(labels_one_hot_format: torch.Tensor) -> torch.Tensor:
    """
    Our task is a binary classification, but we have defined the model's output in one-hot format.
    Sometime we need to change its format to binary format presented by only one number.
    In that case you can use this function

    Args:
        labels_one_hot_format (torch.Tensor): in the shape of (batch_size, 2)

    Returns:
        torch.Tensor: in the shape of (batch_size, )
    """
    assert (labels_one_hot_format.dim() == 2), logger.error("labels must be a tensor having 2 dimensions")
    assert (labels_one_hot_format.shape[1] == 2), logger.error("labels must be in the shape (batch_size, 2)")
    # it is equalt to first columns, all rows
    labels_binary_format = labels_one_hot_format[..., 0]
    return labels_binary_format

def change_labels_from_probablity_format_to_deterministic_format(
        labels_probabilty_format: torch.Tensor,
        is_one_hot_format: bool,
        threshold: float=0.5) -> torch.Tensor:
    """
    After the prediction, we would want to know which label the input image belongs based on a threshold.
    If the probability of fist row is more than threshold, its value will change to one, 
    otherwise it will change to zero, 
    Note that in the one-hot format, the first index shows probality of class valid, and
    the second index shows probabilty of class invalid. 
    Args:
        labels_probabilty_format (torch.Tensor): in the shape of (batch_size, 2) or (batch_size).
                                                 it's elements are between zero and one. 
        threshold (float, optional): It determines the boundary for change the probabilty to one or zero.

    Returns:
        torch.Tensor: in the shape of same as labesl_probabilty_format. Its elements are zero or one
    """
    labesl_deterministic_format = torch.zeros_like(labels_probabilty_format)
    if is_one_hot_format:
        assert (labels_probabilty_format.dim() == 2), logger.error(
            "labels must be a tensor having 2 dimensions if your passed argument for is_one_hot_format is True")
        assert (labels_probabilty_format.shape[1] == 2), logger.error(
            "labels must be in the shape (batch_size, 2) if your passed argument for is_one_hot_format is True")
        mask = labels_probabilty_format[..., 0] >= threshold
        labesl_deterministic_format[..., 0][mask] = 1
        labesl_deterministic_format[..., 1][~mask] = 1
    else:
        assert (labels_probabilty_format.dim() == 1), logger.error(
            "labels must be a tensor having 1 dimension if your passed argument for is_one_hot_format is False")
        mask = labels_probabilty_format >= threshold
        labesl_deterministic_format[mask] = 1
    
    return labesl_deterministic_format


def calculate_classification_metrics(labels_grnd_truth: torch.Tensor, 
                                     labels_predicted: torch.Tensor) -> Dict[str, float]:
    """
    The inputs must be in determisntic format not probabilty format.

    Args:
        labels_grnd_truth (torch.Tensor): must be in the shape of (batch_size)
        labels_predicted (torch.Tensor): must be in the shape of (batch_size)

    Returns:
        Dict[str, float]: it consists of accuracy, precision, recall, f1_score
    """

    assert(labels_grnd_truth.dim() == 1), logger.error(
            "labels_grnd_truth must be a tensor having 1 dimension in the shape the batch_size")
    assert(labels_predicted.dim() == 1), logger.error(
            "labels_predicted must be a tensor having 1 dimension in the shape the batch_size")
    accuracy = accuracy_score(labels_grnd_truth, labels_predicted)
    precision = precision_score(labels_grnd_truth, labels_predicted)
    recall = recall_score(labels_grnd_truth, labels_predicted)
    f1 = f1_score(labels_grnd_truth, labels_predicted)

    return {"accuracy": accuracy, 
            "precision": precision,
            "recall": recall,
            "f1_score": f1}

def plot_confusion_matrix(labels_grnd_truth: torch.Tensor, 
                          labels_predicted: torch.Tensor) -> plt.figure:
    """
    The inputs must be in determinstic format not probability format.
    This function shows the number of true and wrong predictions for each class.
    It is Unnormalized confuction matrix that means the values are in number-format not in percent-format

    Args:
        labels_grnd_truth (torch.Tensor): must be in shape of (batch_size, )
        labels_predicted (torch.Tensor): must be in shape of (batch_size, ) and in determinsitc fromat

    Returns:
        plt.figure: shows how many number of class i has been detected as class j
    """
    assert(labels_grnd_truth.dim() == 1), logger.error(
            "labels_grnd_truth must be a tensor having 1 dimension in the shape the batch_size")
    assert(labels_predicted.dim() == 1), logger.error(
            "labels_predicted must be a tensor having 1 dimension in the shape the batch_size")
    confusion_mat = confusion_matrix(labels_grnd_truth, labels_predicted)

    fig = plt.figure()
    plt.imshow(confusion_mat, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()

    labels_index = [0, 1]
    labels_name = [LABELS_INT_To_STR[i] for i in labels_index]
    plt.xticks(labels_index, labels_name)
    plt.yticks(labels_index, labels_name)
    
    color_threshold = confusion_mat.max() / 2
    # note that the element (i, j) of confusion_mat shows 
    # number of samples from a class whose label value is i, but 
    # detected as a class whose label value is j.
    for i, j in np.ndindex(confusion_mat.shape):
        plt.text(j, i, format(int(confusion_mat[i, j]), "d"), 
                 horizontalalignment="center", 
                 color="white" if confusion_mat[i, j] > color_threshold else "black")

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    return fig

def plot_ROC_curve(labels_grnd_truth: torch.Tensor, 
                   labels_predicted: torch.Tensor) -> plt.figure:
    """
    It takes the true and predicted labels in probability format, and
    plot the receiver operating characterestic curve along side the area under curve (AUC).
    ROC curve plots true positive rate (TPR) against the false positive rate (FPR) at various classification thresholds.
    The more AUC the better classifier.

    Args:
        labels_grnd_truth (torch.Tensor): must be in the shape of (batch_size, )
        labels_predicted (torch.Tensor): must be in the shape of (batch_size, ) and in probabilty format

    Returns:
        plt.figure: show TPR vs FPR
    """
    assert(labels_grnd_truth.dim() == 1), logger.error(
            "labels_grnd_truth must be a tensor having 1 dimension in the shape the batch_size")
    assert(labels_predicted.dim() == 1), logger.error(
            "labels_predicted must be a tensor having 1 dimension in the shape the batch_size")

    # calculate false positive rate, true positive rate and thresholds
    fpr, tpr, thresholds = roc_curve(labels_grnd_truth, labels_predicted)
    # calculate area under the curve (AUC)
    roc_auc = auc(fpr, tpr)

    fig = plt.figure()
    plt.plot(fpr, tpr)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Receiver Operating Characteristic (ROC) (AUC = {roc_auc: 0.2f})")
    
    return fig

def plot_precision_recall_curve(labels_grnd_truth: torch.Tensor, 
                                labels_predicted: torch.Tensor) -> plt.figure:
    """
    It takes the true and predicted labels in probability format, and
    plots the fraction of true positive predictions among all positive predictions vs 
    the fraction of true positive predictions among all actual positive cases.
    It also calculated the area under curve which is called average precision (AP)
    The more AP the better classifier.

    Args:
        labels_grnd_truth (torch.Tensor): must be in the shape of (batch_size, )
        labels_predicted (torch.Tensor): must be in the shape of (batch_size, ) and in probability format

    Returns:
        plt.figure: show the precision of classifying valid images vs the recall of classifying valid images
    """
    assert(labels_grnd_truth.dim() == 1), logger.error(
            "labels_grnd_truth must be a tensor having 1 dimension in the shape the batch_size")
    assert(labels_predicted.dim() == 1), logger.error(
            "labels_predicted must be a tensor having 1 dimension in the shape the batch_size")
    # calculate precision, recall, and thresholds
    precision, recall, thresholds = precision_recall_curve(labels_grnd_truth, labels_predicted)
    # calculate area under the curve (AUC)
    precision_recall_auc = auc(recall, precision)

    fig = plt.figure()
    # plot precision-recall curve
    plt.plot(recall, precision)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve (AP = {precision_recall_auc: 0.2f})")

    return fig

