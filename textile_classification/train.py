"""
Use this python file to train your model.
It reads the existed train.csv and validation.csv files to train the model
Note that you have to pass a .cfg file as the a argument when you want to run this file.
That .cfg file must contain required information for trainin the model.
Pleae take a look at train_input_info.cfg as a template.
You can create your own .cfg file with another name or just change the content of train_input_info.cfg

This is an example how you can train the model:

python .\textile_classification\train.py .\textile_classification\train_input_info.cfg
"""

import pdb
import pretty_errors
import os
import platform
import torch
from torch import nn
from torch.utils.data import DataLoader
import pandas as pd
import argparse
import configparser
from pathlib import Path
import mlflow
from config import DATA_DIR, logger, BASE_DIR
from textile_classification.data_setters import (DatasetCreator, 
                                                 change_images_path_in_train_val_test_csv_files, 
                                                 create_csv_files_for_train_val_test_phases,
                                                 create_csv_files_containing_both_true_and_predicted_labels)
from textile_classification.models import (PretrainedDenseNet, 
                                           PretrainedEfficientNet, 
                                           PretrainedResNeXt)
from textile_classification.utils import (set_seeds, 
                                          get_default_device, 
                                          log_after_feeding_each_batch, 
                                          log_after_ending_each_epoch,
                                          log_overal_performance_afer_ending_all_epoch,
                                          change_labels_from_one_hot_fromat_to_binary_format,
                                          change_labels_from_probablity_format_to_deterministic_format)

def train_model(epochs_num: int, model: nn.Module,
                train_data_loader: DataLoader, val_data_loader: DataLoader,
                loss_function, optimizer, lr_scheduler=None):
    """
    This function is responsible for training the model. 
    This function uses mlflow to log parameters, metrics, checkpoint, figures, and etc in the mlruns directory

    Args:
        epochs_num (int): determines the number of iterations on whole data set.
        model (nn.Module): It also can has been loaded from a checkpoint
        train_data_loader (DataLoader): the data loader which is reposnbile for choosing batches from train's dataset
        val_data_loader (DataLoader): the data loader which is reposnbile for choosing batches from validation's dataset
        loss_function (_type_): is responsible for calculating the gradients of model's parameter
        optimizer (_type_): is responsbile for updating model's parameters based on their gradient
        lr_scheduler (_type_, optional): Is responsible for change learning rate
    """
    
    
    ds_train_size = len(train_data_loader.dataset)
    ds_val_size = len(val_data_loader.dataset)
    # we check validation loss to see if it gets better, 
    # then we save it as a checkpoint
    best_loss_val = 10
    for epoch_idx in range(epochs_num):
        logger.info(f"Epoch {epoch_idx+1}\n-------------------------------")
        
        # Training Phase
        model.train()
        losses_train_epoch = []
        labels_grnd_truth_train_epoch = []
        labels_predicted_train_epoch = []
        image_pathes_train_epoch = []
        
        for batch_idx, batch in enumerate(train_data_loader):
            optimizer.zero_grad()
            images_batch = batch["image"]
            labels_grnd_truth_batch = batch["label_grnd_truth"]
            labels_predicted_batch = model(images_batch)
            # the order of input argument for loss is very important
            loss_batch = loss_function(labels_predicted_batch, labels_grnd_truth_batch)
            loss_batch.backward()
            optimizer.step()
            # convert a torch tensor having only one element to a ordinary python number.
            loss_batch = loss_batch.item()
            # print train loss every 25 batchs in info.log file
            if batch_idx % 25 == 0:
                ds_train_size_so_far = batch_idx * train_data_loader.batch_size
                logger.info(
                    f"train_loss: {loss_batch:0.5f}, num_feeded_data/num_whole_data: [{ds_train_size_so_far}/{ds_train_size}]"
                    )

            # we need to detach labels from gradient and send them on cpu, so numpy and scikit-learn could handle it.
            # since we are done with calculating loss, we don't need to worry about it.
            # we do the same thing for images in inside the log function since
            # we don't need to log images every batch or every epoch, so we don't increase required time
            labels_grnd_truth_batch = labels_grnd_truth_batch.to("cpu")
            labels_predicted_batch = labels_predicted_batch.detach().to("cpu")
            log_after_feeding_each_batch("train", 
                                         epoch_idx=epoch_idx, 
                                         batch_idx=batch_idx,
                                         data_set_size=ds_train_size, 
                                         images=images_batch,
                                         labels_grnd_truth=labels_grnd_truth_batch,
                                         labels_predicted=labels_predicted_batch,
                                         loss=loss_batch, 
                                         epoch_frequency_image_log=50)                

            losses_train_epoch.append(loss_batch)
            labels_grnd_truth_train_epoch.append(labels_grnd_truth_batch)
            labels_predicted_train_epoch.append(labels_predicted_batch)
            image_pathes_train_epoch.extend([image_path for image_path in batch["image_path"]])
        
        # here is the end of (for batch_idx, batch in enumerate(train_data_loader))
            
        log_after_ending_each_epoch(phase="train", epoch_idx=epoch_idx, 
                                    labels_grnd_truth=labels_grnd_truth_train_epoch,
                                    labels_predicted=labels_predicted_train_epoch, 
                                    losses=losses_train_epoch)

        # Validation Phase
        model.eval()
        losses_val_epoch = []
        labels_grnd_truth_val_epoch = []
        labels_predicted_val_epoch = []
        image_pathes_val_epoch = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_data_loader):
                images_batch = batch["image"]
                labels_grnd_truth_batch = batch["label_grnd_truth"]
                labels_predicted_batch = model(images_batch)
                # the order of input argument for loss is very important
                loss_batch = loss_function(labels_predicted_batch, labels_grnd_truth_batch)
                # convert a torch tensor having only one element to a ordinary python number.
                loss_batch = loss_batch.item()
                
                # print validation loss every 25 batchs in info.log file
                if batch_idx % 25 == 0:
                    ds_val_size_so_far = batch_idx * val_data_loader.batch_size
                    logger.info(
                        f"validation_loss: {loss_batch:0.5f}, num_feeded_data/num_whole_data: [{ds_val_size_so_far:d}/{ds_val_size:d}]")
                
                # we need to send labels on cpu, so numpy and scikit-learn could handle it.
                # since we are done with calculating loss, we don't need to worry about it.
                # we do the same thing for images in inside the log function since
                # we don't need to log images every batch or every epoch, so we don't increase required time
                labels_grnd_truth_batch = labels_grnd_truth_batch.to("cpu")
                labels_predicted_batch = labels_predicted_batch.to("cpu")
                log_after_feeding_each_batch("validation", 
                                            epoch_idx=epoch_idx, 
                                            batch_idx=batch_idx,
                                            data_set_size=ds_val_size, 
                                            images=images_batch,
                                            labels_grnd_truth=labels_grnd_truth_batch,
                                            labels_predicted=labels_predicted_batch,
                                            loss=loss_batch, 
                                            epoch_frequency_image_log=50)
                
                losses_val_epoch.append(loss_batch)
                labels_grnd_truth_val_epoch.append(labels_grnd_truth_batch)
                labels_predicted_val_epoch.append(labels_predicted_batch)
                image_pathes_val_epoch.extend([image_path for image_path in batch["image_path"]])
            
            # here is the end of (for batch_idx, batch in enumerate(val_data_loader))
        # here is the end of (with torch.no_grad())
        
        log_after_ending_each_epoch(phase= "validation", epoch_idx=epoch_idx, 
                                    labels_grnd_truth=labels_grnd_truth_val_epoch,
                                    labels_predicted=labels_predicted_val_epoch, 
                                    losses=losses_val_epoch)
        
        # Record and Update Learning Rate
        if lr_scheduler is not None:
            lr_scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            # log learing rate
            mlflow.log_metric("train/per_epoch/learning_rate", current_lr, epoch_idx)
            
        # check the validation loss evey 50 ephoc to see if the model perform better or not,
        # if it is better than previous check, we will save it a checkpoint.
        if epoch_idx % 50 == 0:
            loss_val_mean = torch.mean(torch.tensor(losses_val_epoch)).item()
            if loss_val_mean < best_loss_val:
                best_loss_val = loss_val_mean

                loss_train_mean = torch.mean(torch.tensor(losses_train_epoch)).item()

                # save the best checkpoint to continue training later if it was necessay.
                # also for using it in test and prediction phase
                # we almost save everthing to be able of retrieving the condistion of the last checkpoint
                # we also need to need to save ds_train to use the same transformation for testing
                checkpoint = {
                    "loss": loss_train_mean,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "lr_scheduler_state_dict": lr_scheduler.state_dict(),
                    "data_set": ds_train,
                    "loss_function": loss_function,
                    "model_type": type(model).__name__}
                
                checkpoint_path = BASE_DIR.joinpath("checkpoint.pt")
                torch.save(checkpoint, checkpoint_path)
                mlflow.log_artifact(checkpoint_path, "saved_checkpoints/best_checkpoint")
                # after loggin the checlpoint in mlrun, 
                # we remove its source to avoid keeping same files at different loation
                os.remove(checkpoint_path)

                log_overal_performance_afer_ending_all_epoch(
                    phase="train", 
                    labels_grnd_truth=labels_grnd_truth_train_epoch,
                    labels_predicted=labels_predicted_train_epoch,
                    losses=losses_train_epoch)
                
                log_overal_performance_afer_ending_all_epoch(
                    phase="validation", 
                    labels_grnd_truth=labels_grnd_truth_val_epoch,
                    labels_predicted=labels_predicted_val_epoch,
                    losses=losses_val_epoch)
                
                # create csv files for the best checkpoint to see 
                # which images is classified correctly and which one is not.
                for csv_name, image_pathes, labels_grnd_truth, labels_predicted in zip(
                    ["train.csv", "validation.csv"],
                    [image_pathes_train_epoch, image_pathes_val_epoch], 
                    [labels_grnd_truth_train_epoch, labels_grnd_truth_val_epoch],
                    [labels_predicted_train_epoch, labels_predicted_val_epoch]):
                        
                        labels_grnd_truth = torch.cat(labels_grnd_truth, dim=0)
                        labels_predicted = torch.cat(labels_predicted, dim=0)
                        
                        labels_grnd_truth = change_labels_from_one_hot_fromat_to_binary_format(
                            labels_grnd_truth).tolist()
                        labels_predicted = change_labels_from_one_hot_fromat_to_binary_format(
                            labels_predicted)
                        
                        labels_predicted = change_labels_from_probablity_format_to_deterministic_format(
                            labels_predicted, 
                            is_one_hot_format=False).tolist()
                        
                        create_csv_files_containing_both_true_and_predicted_labels(
                            csv_name=csv_name,
                            image_pathes=image_pathes,
                            labels_grund_truth=labels_grnd_truth, 
                            labels_predicted=labels_predicted,
                            csv_dir=BASE_DIR)
                        
                        csv_path = BASE_DIR.joinpath(csv_name)
                        mlflow.log_artifact(csv_path, "csv_files")
                        # after loggin the csv file in mlrun, 
                        # we remove its source to avoid keeping same files at different loation
                        os.remove(csv_path)
            # end of (if loss_val_mean < best_loss_val:)
                    
    # here is the end of (epoch_idx in range(epochs_num))
        
    logger.info("Training is Done")

if __name__ == "__main__":

    seed = 12345
    set_seeds(seed)

    # make the train.py accept argument via the command-line.
    # it is named input_info, and its positional and mandatory. 
    parser = argparse.ArgumentParser(
        description=(
        ("Train the model based on the information provided in the input config file." 
         " Please take a look at train_input_info.cfg file as an example of the input file.")
        ))
    parser.add_argument(
        "input_info_file", 
        help="the path of .cfg file containing all information for training"
        )
    
    args = parser.parse_args()
    input_info_file = Path(args.input_info_file)
    assert(input_info_file.exists()), logger.error(f"File {input_info_file} does not exist.")

    config = configparser.ConfigParser()
    config.read(input_info_file)

    max_epochs = int(config["HyperParameters"]["number_of_epochs"])
    batch_size = int(config["HyperParameters"]["batch_size"])
    lr_start = float(config["HyperParameters"]["lr_start"])
    lr_gamma_decay = float(config["HyperParameters"]["lr_gamma_decay"])
    lr_stepsize_decay = int(config["HyperParameters"]["lr_stepsize_decay"])
    selected_model = (config["HyperParameters"]["selected_model"]).lower()
    assert(selected_model in ["efficientnet", "resnext", "densenet"]), logger.error(
        "selected model is not among available models")
    experiment_name = config["ExperimentIDs"]["experiment_name"]
    if experiment_name.lower() == "none":
        experiment_name = None
    checkpoint_file = config["Checkpoint"]["checkpoint_file"]
    if checkpoint_file.lower() == "none":
        checkpoint_file = None
    else:
        assert(Path(checkpoint_file).exists()), logger.error(f"File {checkpoint_file} does not exist.")
    
    dtype = torch.float32
    device = get_default_device()

    change_images_path_in_train_val_test_csv_files(csv_dir=DATA_DIR)
    csv_train = DATA_DIR.joinpath("train.csv")
    csv_val = DATA_DIR.joinpath("validation.csv")

    df_train = pd.read_csv(csv_train)
    df_val = pd.read_csv(csv_val)
    ds_train = DatasetCreator(df_train, device=device, dtype=dtype)
    # all input arguments for creating ds_val must be the same as ds_train's except the data_frame.
    ds_val = DatasetCreator(df_val, input_transform=ds_train.input_transform, 
                            device=ds_train.device, dtype=ds_train.dtype)
    
    # we need to shuffle ds_train after each epoch, but
    # it is not necessary for validation data
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=batch_size)

    # first create temporary model to calculate num_features extraced by the CNN layers of pre-trained model.
    # we use that number to change the FC layer based on our task
    # we need to create a batch of our data_set to be able of using that function, and
    # finally we can create our new model
    # don't forget to move the model to the chosen device
    batch_example = next(iter(dl_train))
    input_example = (batch_example["image"]).to("cpu")
    if selected_model == "efficientnet":
        model = PretrainedEfficientNet()
        num_features = model.find_num_feaures_extraced_by_cnn_layers(input_example)
        model = PretrainedEfficientNet(new_in_fearues_of_fc=num_features, new_out_features_of_fc=2)
    elif selected_model == "resnext":
        model = PretrainedResNeXt()
        num_features = model.find_num_feaures_extraced_by_cnn_layers(input_example)
        model = PretrainedResNeXt(new_in_fearues_of_fc=num_features, new_out_features_of_fc=2)
    else:
        model = PretrainedDenseNet()
        num_features = model.find_num_feaures_extraced_by_cnn_layers(input_example)
        model = PretrainedDenseNet(new_in_fearues_of_fc=num_features, new_out_features_of_fc=2)
    
    # move the model to device
    model.to(device=device)

    # becuase the data_set is imbalanced we need to consider different weights for each label in the loss function.
    # the label with less number has more weight.
    # it also must be on the same device as the data_set and model
    weight = torch.tensor([ds_train.num_images_per_label["invalid"] / len(ds_train), 
                           ds_train.num_images_per_label["valid"] / len(ds_train)], 
                           dtype=dtype, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_start)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_stepsize_decay, gamma=lr_gamma_decay)
    # if we want to continue training from a previous saved checkpoint, 
    # we need to load the checkpoint and use it.
    if checkpoint_file is not None:
        checkpoint = torch.load(Path(checkpoint_file))
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
    
    # start logging, uri syntax is different for different operating system.
    tracking_folder = str(BASE_DIR.joinpath("mlruns"))
    os_name = platform.system()
    if os_name == "Windows":
        tracking_uri = f"file:///{tracking_folder}"
    else:
        tracking_uri = f"file:/{tracking_folder}"

    mlflow.set_tracking_uri(tracking_uri)
    # create a new experiment or set it to existed one.
    if experiment_name is not None:
        mlflow.set_experiment(experiment_name=experiment_name)
    with mlflow.start_run():
        mlflow.log_params({"epochs_num": max_epochs,
                           "batch_size": batch_size,
                           "lr_start": lr_start,
                           "lr_gama_decay": lr_gamma_decay,
                           "lr_step_size_decay": lr_stepsize_decay,
                           "selected_model": selected_model})
        if checkpoint_file is not None:
            mlflow.log_artifact(Path(checkpoint_file), 
                                "saved_checkpoints/starting_checkpoint")
    
        train_model(max_epochs, model, dl_train, dl_val, 
                    loss_fn, optimizer, lr_scheduler)

