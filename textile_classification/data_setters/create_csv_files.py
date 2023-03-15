
from pathlib import Path
from typing import List, Union
import math
import pandas as pd
from config import logger, DATA_DIR, BASE_DIR
import re

def create_csv_files_for_train_val_test_phases(
        images_parent_dir: Union[str, Path] = DATA_DIR,
        extensions: List[str] = ["png", "jpg", "jpeg", "jpe", "jfif"],
        portion_train: float = 0.8, portion_val: float = 0.15,
        csv_dir: Union[str, Path] = DATA_DIR
        ):
    """
    This function search in these two directories (data/valid, data/nonvalid) to find images and their label, 
    and divide them into train, val and test data by creating a csv file for each of them.
    We will need these csv files to create training, validation and testing dataset in the future. 
    Note: portion_train + portion_val must be less than or equal to one.
    

    Args:
        images_parent_dir (Union[str, Path], optional): the direcory you can see these folders (valid and invalid).
        extensions (List[str], optional): The availabe extensions for the image files.
        portion_train (float, optional): The portion of number of trainind data to number of whole data.
        portion_val (float, optional): The portion of number of validation data to number of whole data.
        csv_dir (Union[str, Path], optional): the directory you want to save csv_files.
    """

    # make sure you are passing correct values for input arguments
    # the size of training data must be bigger than validation data
    assert (0.7 <= portion_train < 1.0), logger.error(
             "training data has not accpetable portion of the whole data")
        
    assert (0.0 <= portion_val <= 0.3), logger.error(
            "validation data has not accpetable portion of the whole data")
    
    assert ((portion_train + portion_val) <= 1.0), logger.error(
            "the number of training and validation data must not bigger than the whole data")
    
    # Search for all image files for each label and divide them into train, val and test . 
    df_dict_train = {"image_path": [], "label": []}
    df_dict_val = {"image_path": [], "label": []}
    df_dict_test = {"image_path": [], "label": []}
    for label in ["valid", "invalid"]:
        images_sub_dir = images_parent_dir.joinpath(label)
        image_paths = []
        for each_extension in extensions:
            image_paths.extend([x for x in Path(images_sub_dir).glob(f"*.{each_extension}")])
        
        num_images = len(image_paths)

        for i, each_image_path in enumerate(image_paths):
            if i < math.floor(portion_train * num_images):
                df_dict_train["image_path"].append(each_image_path)
                df_dict_train["label"].append(label)
            elif i < math.floor((portion_train + portion_val) * num_images):
                df_dict_val["image_path"].append(each_image_path)
                df_dict_val["label"].append(label)
            else:
                df_dict_test["image_path"].append(each_image_path)
                df_dict_test["label"].append(label)
    
    # create csv file for each data_frame
    df_pd_train = pd.DataFrame(df_dict_train)
    csv_path_train = csv_dir.joinpath("train.csv")
    df_pd_train.to_csv(csv_path_train, index=False)

    df_pd_val = pd.DataFrame(df_dict_val)
    csv_path_val = csv_dir.joinpath("validation.csv")
    df_pd_val.to_csv(csv_path_val, index=False)

    df_pd_test = pd.DataFrame(df_dict_test)
    csv_path_test = csv_dir.joinpath("test.csv")
    df_pd_test.to_csv(csv_path_test, index=False)

def create_csv_file_for_prediction_phase(
        images_dir: Union[str, Path],
        extensions: List[str] = ["png", "jpg", "jpeg", "jpe", "jfif"]):
    """
    The full path directory for images you want to predict their label.

    Args:
        images_dir (Union[str, Path]): The full path directory for images you want to predict their label.
        extensions (List[str], optional): These kind of images will be found and put into the csv file
    """
    image_paths = []
    df_dict = {"image_path": [], "label": []}
    for each_extension in extensions:
        image_paths.extend([x for x in Path(images_dir).glob(f"*.{each_extension}")])
    
    for each_image_path in image_paths:
        df_dict["image_path"].append(each_image_path)
        df_dict["label"].append("unknown")
    
    # create csv file for each data_frame
    df_pd = pd.DataFrame(df_dict)
    csv_path = Path(images_dir).joinpath("prediction.csv")
    df_pd.to_csv(csv_path, index=False)

def change_images_path_in_train_val_test_csv_files(csv_dir: Union[str, Path] = DATA_DIR):
    """
    The csv files contain absolute pathes for image files.
    Those are not going to be correct on other computer, so
    we need to change them if we are going to use the csv files have been created on another computer

    Args:
        csv_dir (Union[str, Path], optional): the direcoty you are seeing the csc files.
                                              the new csv files are going to be save there and replace the old ones.
    """
    
    for csv_name in ["train.csv", "validation.csv", "test.csv"]:
        csv_path = Path(csv_dir).joinpath(csv_name)
        df = pd.read_csv(csv_path)
        for idx in range(len(df)):
            old_path = Path(df.loc[idx, "image_path"])
            new_path = Path(csv_dir).joinpath(old_path.parent.name, old_path.name)
            df.loc[idx, "image_path"] = new_path
        df.to_csv(csv_path, index=False)

def create_csv_files_containing_both_true_and_predicted_labels(csv_name: str, 
                                                               image_pathes: List[str],
                                                               labels_grund_truth: List[float],
                                                               labels_predicted: List[float],
                                                               threshold: float=0.5,
                                                               csv_dir: Union[str, Path]=BASE_DIR) -> None:
    """
    After classfication use this function to create a csv file.
    It shows you which image_path has been classified correctly, and which one is not.
    It helps you to have a better understanding which kind of images the model is wrongly classifying.
    You can use this knowledge to find a way like data augmentation to increase the model's performance.

    Args:
        csv_name (str): It must be train, validation or test
        image_pathes (List[str]): _description_
        labels_grund_truth (List[float]): _description_
        labels_predicted (List[float]): it must be in probability format
        threshold (float, optional): It should be between zero and one.
                                     if the probability of the prediced label is more than threshold, 
                                     the image is considered as valid image, otherwise it is invalid.
        csv_dir (Union[str, Path], optional): the directroy you want to create the csv file
    """
    assert (csv_name in ["train.csv", "validation.csv", "test.csv"]), logger.error(
        f"csv_name({csv_name}) must be train.csv, validation.csv or test.csv")
    assert (len(image_pathes) == len(labels_grund_truth)), logger.error(
        "image_pathes and labels_grund_truth must have the same length")
    assert (len(labels_predicted) == len(labels_grund_truth)), logger.error(
        "labels_predicted and labels_grund_truth must have the same length")
    
    df_dict = dict()
    df_dict["image_path"] = image_pathes
    df_dict["label_grnd_truth"] = ["valid" if (x == 1) else "invalid" for x in labels_grund_truth]
    df_dict["label_predicted"] = ["valid" if (x >= threshold) else "invalid" for x in labels_predicted]
    df_pd = pd.DataFrame(df_dict)
    csv_path = csv_dir.joinpath(csv_name)
    df_pd.to_csv(csv_path, index=False)
