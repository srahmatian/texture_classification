"""
Use this python file to predict the labels of unseen images using a trained model.
It classifies the images located in the your provided directroy file, and
separate them into subfolders named valid and invalid based on their predicted labels.
Use this file when you don't know the ground truths for the labes.
Note that you have to pass a .cfg file as the a argument when you want to run this file.
That .cfg file must contain required information for the prediction like images_dir and the checkpoint of trained model.
Pleae take a look at predict_input_info.cfg as a template.
You can create your own .cfg file with another name or just change the content of predict_input_info.cfg
This is an example how you can test the model:

python .\textile_classification\predict.py .\textile_classification\predict_input_info.cfg
"""

import pdb
import pretty_errors
import shutil
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import Dataset
import argparse
import configparser
import pandas as pd
from config import logger

from textile_classification.utils import (get_default_device, 
                                          change_labels_from_one_hot_fromat_to_binary_format, 
                                          change_labels_from_probablity_format_to_deterministic_format)
from textile_classification.data_setters import (DatasetCreator, 
                                                 create_csv_file_for_prediction_phase)
from textile_classification.models import (PretrainedDenseNet, 
                                           PretrainedEfficientNet, 
                                           PretrainedResNeXt)

def predict_model(model: nn.Module, ds_prediction: Dataset, threshold: float=0.5):
    """
    Use this function to test the performance of the model on unseen images, and
    you don't know their true labes.
    So there is no metric calculation here. 
    This function create two subfolders (valid, invalid) in the directory of input images, and 
    put them there based on its prediction.
    It also create a csv file name prediction.csv and store it there.


    Args:
        model (nn.Module): The model that is loaded by a checkoint
        ds_prediction (Dataset): We don't need to pass data loader since we pass the images one by one
        threshold (float, optional): It should be between zero and one.
                                     if the probability of the prediced label is more than threshold, 
                                     the image is considered as valid image, otherwise it is invalid. 
    """
    
    # fot testing the model, we need to remove dropout, fix batch_normalization, and etc using eval()
    model.eval()
    with torch.no_grad():
        labels_predicted_all = []
        ds_predict_size = len(ds_prediction)
        for idx in range(ds_predict_size):
            # note that targets are nan in predication phase
            input_target = ds_prediction[idx]
            # data_set returns tensor without having a dimension related to batch, so
            # to feed the data into model, we need to unsqueeze them.
            image = input_target["image"].unsqueeze(dim=0)
            label_predicted = model(image)
            labels_predicted_all.append(label_predicted)
            if idx % 25 == 0:
                logger.info(f"num_feeded_data/num_whole_data: [{idx}/{ds_predict_size}]")
        
        labels_predicted_all = torch.cat(labels_predicted_all, dim=0)
        labels_predicted_all = change_labels_from_one_hot_fromat_to_binary_format(labels_predicted_all)
        labels_predicted_all = change_labels_from_probablity_format_to_deterministic_format(
            labels_predicted_all, 
            is_one_hot_format=False, 
            threshold=threshold)
        
        # separate images intp valid and invalid folders
        images_dir = Path(ds_prediction.data_frame_pd.loc[0, "image_path"]).parent

        valid_dir = images_dir.joinpath("valid")
        valid_dir.mkdir(parents=True, exist_ok=True)

        invalid_dir = images_dir.joinpath("invalid") 
        invalid_dir.mkdir(parents=True, exist_ok=True)

        for idx in range(ds_predict_size):
            if labels_predicted_all[idx] == 1:
                ds_prediction.data_frame_pd.loc[idx, "label"] = "valid"
                image_path = Path(ds_prediction.data_frame_pd.loc[idx, "image_path"])
                shutil.move(image_path, valid_dir)
                ds_prediction.data_frame_pd.loc[idx, "image_path"] = valid_dir.joinpath(image_path.name)
            else:
                ds_prediction.data_frame_pd.loc[idx, "label"] = "invalid"
                image_path = Path(ds_prediction.data_frame_pd.loc[idx, "image_path"])
                shutil.move(image_path, invalid_dir)
                ds_prediction.data_frame_pd.loc[idx, "image_path"] = invalid_dir.joinpath(image_path.name)
        
        # create csv file for the data_frame
        csv_path = images_dir.joinpath("prediction.csv")
        ds_prediction.data_frame_pd.to_csv(csv_path, index=False)
    
    logger.info("Prediction is Done")

if __name__ == "__main__":
    
    # make the predict.py accept argument via the command-line.
    # it is named input_info, and its positional and mandatory. 
    parser = argparse.ArgumentParser(
        description=(
        ("Classify images with unkown labels by a trained model." 
        " Please take a look at predict_input_info.cfg file as an example of the input file.")
        ))
    parser.add_argument(
        "input_info_file", 
        help="the path of .cfg file containing all information for prediction"
        )
    args = parser.parse_args()
    input_info_file = Path(args.input_info_file)
    assert(input_info_file.exists()), logger.error(f"File {input_info_file} does not exist.")
    
    config = configparser.ConfigParser()
    config.read(input_info_file)

    threshold = float(config["RequiredInfo"]["threshold"])

    checkpoint_file = config["RequiredInfo"]["checkpoint_file"]
    assert(Path(checkpoint_file).exists()), logger.error(f"File {checkpoint_file} does not exist.")
    
    images_dir = Path(config["RequiredInfo"]["images_dir"])
    assert(images_dir.exists()), logger.error(f"Directory {images_dir} does not exist.")
    # create prediction.csv file
    create_csv_file_for_prediction_phase(images_dir=images_dir)
    df_prediction = pd.read_csv(images_dir.joinpath("prediction.csv"))

    # we might train model on another device, but we might need to test it on another one. 
    device = get_default_device()
    checkpoint = torch.load(Path(checkpoint_file), map_location=device)
    ds_train = checkpoint["data_set"]

    # all input arguments for creating ds_prediction must be the same as ds_train's except the data_frame.
    ds_prediction = DatasetCreator(df_prediction, input_transform=ds_train.input_transform, 
                                   device=device, dtype=ds_train.dtype)

    model_type = checkpoint["model_type"]
    assert(model_type in ["PretrainedEfficientNet", "PretrainedResNeXt", "PretrainedDenseNet"]), logger.error(
        "selected model is not among available models")
    
    # we need to feed an example to the model to customize it for our task.
    input_example = ds_prediction[0]["image"].to("cpu").unsqueeze(dim=0)
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
    
    predict_model(model, ds_prediction, threshold)
