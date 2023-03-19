"""
Use this python file to test a trained model.
It reads the existed test.csv file to do the prediction.
Use this file when you know the ground truths for the labes.
Note that you have to pass a .cfg file as the a argument when you want to run this file.
That .cfg file must contain required information for testing the model.
Pleae take a look at test_input_info.cfg as a template.
You can create your own .cfg file with another name or just change the content of test_input_info.cfg

This is an example how you can test the model:

python .\textile_classification\test.py .\textile_classification\test_input_info.cfg
"""

import pdb
import pretty_errors
import os
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import Dataset
import argparse
import configparser
import mlflow
import pandas as pd
import platform
from config import BASE_DIR, DATA_DIR, logger
from textile_classification.data_setters import (DatasetCreator, 
                                                 change_images_path_in_train_val_test_csv_files)
from textile_classification.models import (PretrainedDenseNet, 
                                           PretrainedEfficientNet, 
                                           PretrainedResNeXt)
from textile_classification.utils import (get_default_device,
                                          log_after_ending_each_epoch, 
                                          log_overal_performance_afer_ending_all_epoch)

def test_model(model: nn.Module, ds_test: Dataset, loss_function, threshold: float=0.5):
    """
    Use this function to test the performance of the model on unseen images,
    but you still know their true labes.

    This function uses mlflow to log parameters, metrics, figures, and etc in the mlruns directory.
    It also generate a csv file and store it in mlruns. 
    The csv file consists of image_pathes and their corresponding true and predicted labels.

    Args:
        model (nn.Module): The model that is loaded by a checkopoint
        ds_test (Dataset): We don't need to pass data loader since we pass the images one by one
        loss_function (_type_): The same loss function used in training for evaluating the model
        threshold (float, optional): It should be between zero and one.
                                     if the probability of the prediced label is more than threshold, 
                                     the image is considered as valid image, otherwise it is invalid.
    """
    
    # fot testing the model, we need to remove dropout, fix batch_normalization, and etc using eval()
    model.eval()
    with torch.no_grad():
        losses_all = []
        labels_grnd_truth_all = []
        labels_predicted_all = []
        ds_test_size = len(ds_test)
        df_dict = {"image_path": [], "label_grnd_truth": [], "label_predicted": []}
        for idx in range(ds_test_size):
            input_target = ds_test[idx]
            # data_set returns tensor without having a dimension related to batch, so
            # to feed the data into model, we need to unsqueeze them.
            image = input_target["image"].unsqueeze(dim=0)
            label_grnd_truth = input_target["label_grnd_truth"].unsqueeze(dim=0)
            label_predicted = model(image)
            # the order of input argument for loss is very important
            loss = loss_function(label_predicted, label_grnd_truth)
            # convert a torch tensor having only one element to a ordinary python number.
            loss = loss.item()
            if idx % 25 == 0:
                logger.info(
                    f"test_loss: {loss:0.5f}, num_feeded_data/num_whole_data: [{idx}/{ds_test_size}]"
                    )
            losses_all.append(loss)
            labels_grnd_truth_all.append(label_grnd_truth)
            labels_predicted_all.append(label_predicted)

            df_dict["image_path"].append(ds_test.data_frame_pd.loc[idx, "image_path"])
            df_dict["label_grnd_truth"].append(ds_test.data_frame_pd.loc[idx, "label"])
            if label_predicted[0, 0] >= threshold:
                df_dict["label_predicted"].append("valid")
            else:
                df_dict["label_predicted"].append("invalid")
            
        # create a csv file showing true and predicted labels
        df_pd = pd.DataFrame(df_dict)
        csv_path = BASE_DIR.joinpath("test.csv")
        df_pd.to_csv(csv_path, index=False)

        mlflow.log_artifact(csv_path, "csv_files")
        # after loggin the csv file in mlrun, 
        # we remove its source to avoid keeping same files at different loation
        os.remove(csv_path)
        
        log_after_ending_each_epoch(
            phase="test", epoch_idx=0, 
            labels_grnd_truth=labels_grnd_truth_all,
            labels_predicted=labels_predicted_all, 
            losses=losses_all)
        
        log_overal_performance_afer_ending_all_epoch(
            phase="test", 
            labels_grnd_truth=labels_grnd_truth_all,
            labels_predicted=labels_predicted_all,
            losses=losses_all)
    
    logger.info("Testing is Done")

if __name__ == "__main__":
    
    # make the test.py accept argument via the command-line.
    # it is named input_info, and its positional and mandatory. 
    parser = argparse.ArgumentParser(
        description=(
        ("Test a trained model based on the information provided in the input config file." 
        " Please take a look at test_input_info.cfg file as an example of the input file.")
        ))
    parser.add_argument(
        "input_info_file", 
        help="the path of .cfg file containing all information for testing"
        )
    args = parser.parse_args()
    input_info_file = Path(args.input_info_file)
    assert(input_info_file.exists()), logger.error(f"File {input_info_file} does not exist.")

    config = configparser.ConfigParser()
    config.read(input_info_file)

    experiment_name = config["ExperimentIDs"]["experiment_name"]
    if experiment_name.lower() == "none":
        experiment_name = None
    run_id = config["ExperimentIDs"]["run_id"]
    if run_id.lower() == "none":
        run_id = None
    checkpoint_file = config["Checkpoint"]["checkpoint_file"]
    assert(Path(checkpoint_file).exists()), logger.error(f"File {checkpoint_file} does not exist.")

    # we need to change the pathes inside csv files have been creaed on another compurer
    change_images_path_in_train_val_test_csv_files(csv_dir=DATA_DIR)
    df_test = pd.read_csv(DATA_DIR.joinpath("test.csv"))

    # we might train model on another device, but we might need to test it on another one. 
    device = get_default_device()
    checkpoint = torch.load(Path(checkpoint_file), map_location=device)
    ds_train = checkpoint["data_set"]

    # all input arguments for creating ds_test must be the same as ds_train's except the data_frame.
    ds_test = DatasetCreator(df_test, input_transform=ds_train.input_transform, 
                             device=device, dtype=ds_train.dtype)

    model_type = checkpoint["model_type"]
    assert(model_type in ["PretrainedEfficientNet", "PretrainedResNeXt", "PretrainedDenseNet"]), logger.error(
        "selected model is not among available models")
    
    # we need to feed an example to the model to customize it for our task.
    input_example = ds_test[0]["image"].to("cpu").unsqueeze(dim=0)
    if model_type == "PretrainedEfficientNet":
        model = PretrainedEfficientNet()
        num_features = model.find_num_feaures_extraced_by_cnn_layers(input_example)
        model = PretrainedEfficientNet(new_in_fearues_of_fc=num_features, new_out_features_of_fc=2)
    elif model_type == "PretrainedResNeXt":
        model = PretrainedResNeXt()
        num_features = model.find_num_feaures_extraced_by_cnn_layers(input_example)
        model = PretrainedResNeXt(new_in_fearues_of_fc=num_features, new_out_features_of_fc=2)
    else:
        model = PretrainedDenseNet()
        num_features = model.find_num_feaures_extraced_by_cnn_layers(input_example)
        model = PretrainedDenseNet(new_in_fearues_of_fc=num_features, new_out_features_of_fc=2)
    
    model.to(device=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    # start logging, uri syntax is different for different operating system.
    tracking_folder = str(BASE_DIR.joinpath("mlruns"))
    os_name = platform.system()
    if os_name == "Windows":
        tracking_uri = f"file:///{tracking_folder}"
    else:
        tracking_uri = f"file:{tracking_folder}"

    mlflow.set_tracking_uri(tracking_uri)
    # create the experiment to existed one.
    if experiment_name is not None:
        mlflow.set_experiment(experiment_name=experiment_name)
    
    with mlflow.start_run(run_id):
        test_model(model, ds_test, checkpoint["loss_function"])
