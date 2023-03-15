
from typing import Union, List
import torch
import mlflow
from textile_classification.utils import (change_labels_from_one_hot_fromat_to_binary_format,
                                          change_labels_from_probablity_format_to_deterministic_format,
                                          calculate_classification_metrics,
                                          plot_confusion_matrix,
                                          plot_ROC_curve,
                                          plot_precision_recall_curve, 
                                          plot_images)

def log_after_feeding_each_batch(
        phase: str, epoch_idx: int, batch_idx: int, data_set_size: int,
        images: torch.Tensor, 
        labels_grnd_truth: torch.Tensor, 
        labels_predicted: torch.Tensor,
        loss: float, epoch_frequency_image_log: int = 50):
    """
    Using the function, during training, validating or testing the model, we log metrics per batch.
    In input predicted labels must be in probability format

    Args:
        phase (str): for example, it can be train, validation, test
        epoch_idx (int): _description_
        batch_idx (int): _description_
        images (torch.Tensor): must be in the shape of (batch_size, 3, height, width)
        labels_grnd_truth (torch.Tensor): must be in the shape of (batch_size, 2)
        labels_predicted (torch.Tensor): must be in the shape of (batch_size, 2)
        loss (float): loss value abotained from the loss function
        epoch_frequency_image_log (int): we also plot the wrongly classified images to find which images are difficult for model
                                         if (epoch_idx % epoch_frequency_image_log = 0) and (batch_idx == 0)
    """
    labels_grnd_truth = change_labels_from_one_hot_fromat_to_binary_format(labels_grnd_truth)
    labels_predicted = change_labels_from_one_hot_fromat_to_binary_format(labels_predicted)
    labels_predicted = change_labels_from_probablity_format_to_deterministic_format(labels_predicted, 
                                                                                    is_one_hot_format=False)

    metrics = calculate_classification_metrics(labels_grnd_truth, labels_predicted)
    mlflow.log_metrics({f"{phase}/per_batch/loss": loss, 
                        f"{phase}/per_batch/accuracy": metrics["accuracy"],
                        f"{phase}/per_batch/precision": metrics["precision"],
                        f"{phase}/per_batch/recall": metrics["recall"]}, 
                        epoch_idx * data_set_size + batch_idx)
    
    # we can not log images for many times, it takes a lot of memory and time
    if (epoch_idx % epoch_frequency_image_log == 0) and (batch_idx == 0):
        # find indices in the batch with wrong prediction 
        mask = labels_predicted != labels_grnd_truth
        if torch.any(mask) == True:
            # just we choose at most 16 samples of wrong classification
            fig = plot_images(images=(images[mask][0:16]).to("cpu"), 
                              labels_grnd_truth=labels_grnd_truth[mask][0:16], 
                              labels_predicted=labels_predicted[mask][0:16])
            mlflow.log_figure(fig, f"figures/{phase}/wrongly_classified/figure_epoch_{epoch_idx}_batch_{batch_idx}.png")


def log_after_ending_each_epoch(
        phase: str, epoch_idx: int,
        labels_grnd_truth: List[torch.Tensor], 
        labels_predicted: List[torch.Tensor],
        losses: List[float]):
    """
    We use this function to log metric at the end of each epoch. 
    The input predicted labels must be in probability format

    Args:
        phase (str): for example, it can be train, validation or test
        epoch_idx (int): _description_
        labels_grnd_truth (List[torch.Tensor]): list of tensors with shape of (batch_size, 2)
        labels_predicted (List[torch.Tensor]): list of tensors with shape of (batch_size, 2)
        losses (List[float]): list of losses of ordinary python number
    """
    assert isinstance(labels_grnd_truth, list)
    assert isinstance(labels_predicted, list)
    assert isinstance(losses, list)
    labels_grnd_truth = torch.cat(labels_grnd_truth, dim=0)
    labels_predicted = torch.cat(labels_predicted, dim=0)
    labels_grnd_truth = change_labels_from_one_hot_fromat_to_binary_format(labels_grnd_truth)
    labels_predicted = change_labels_from_one_hot_fromat_to_binary_format(labels_predicted)
    labels_predicted = change_labels_from_probablity_format_to_deterministic_format(labels_predicted, 
                                                                                    is_one_hot_format=False)

    metrics = calculate_classification_metrics(labels_grnd_truth, labels_predicted)
    mlflow.log_metrics({f"{phase}/per_epoch/loss": torch.mean(torch.tensor(losses)).item(), 
                        f"{phase}/per_epoch/accuracy": metrics["accuracy"],
                        f"{phase}/per_epoch/precision": metrics["precision"],
                        f"{phase}/per_epoch/recall": metrics["recall"]}, 
                        epoch_idx)

def log_overal_performance_afer_ending_all_epoch(
        phase: str, 
        labels_grnd_truth: List[torch.Tensor], 
        labels_predicted: List[torch.Tensor],
        losses: List[float]):
    
    """
    We use this function to get the final performance of the model. 
    The input predicted labels must be in probability format

    Args:
        phase (str): for example, it can be train, validation or test
        labels_grnd_truth (List[torch.Tensor]): list of tensors with shape of (batch_size, 2)
        labels_predicted (List[torch.Tensor]): list of tensors with shape of (batch_size, 2) in probability format
        losses (List[float]): list of losses of ordinary python number
    """
    assert isinstance(labels_grnd_truth, list)
    assert isinstance(labels_predicted, list)
    assert isinstance(losses, list)
    labels_grnd_truth = torch.cat(labels_grnd_truth, dim=0)
    labels_predicted = torch.cat(labels_predicted, dim=0)
    labels_grnd_truth = change_labels_from_one_hot_fromat_to_binary_format(labels_grnd_truth)
    labels_predicted = change_labels_from_one_hot_fromat_to_binary_format(labels_predicted)

    fig_roc = plot_ROC_curve(labels_grnd_truth.clone(), labels_predicted.clone())
    mlflow.log_figure(fig_roc, f"figures/{phase}/overl_performance/ROC_curve.png")

    fig_precision_recall = plot_precision_recall_curve(labels_grnd_truth.clone(), labels_predicted.clone())
    mlflow.log_figure(fig_precision_recall, f"figures/{phase}/overl_performance/precision_recall_curve.png")


    labels_predicted = change_labels_from_probablity_format_to_deterministic_format(labels_predicted, 
                                                                                    is_one_hot_format=False)

    metrics = calculate_classification_metrics(labels_grnd_truth, labels_predicted)
    mlflow.log_metrics({f"{phase}/overl_performance/loss": torch.mean(torch.tensor(losses)).item(), 
                        f"{phase}/overl_performance/accuracy": metrics["accuracy"],
                        f"{phase}/overl_performance/precision": metrics["precision"],
                        f"{phase}/overl_performance/recall": metrics["recall"],
                        f"{phase}/overl_performance/f1_score": metrics["f1_score"]})
    fig_confusion_mat = plot_confusion_matrix(labels_grnd_truth.clone(), labels_predicted.clone())
    mlflow.log_figure(fig_confusion_mat, f"figures/{phase}/overl_performance/confusion_matrix.png")
    
