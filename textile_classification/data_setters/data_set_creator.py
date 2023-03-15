
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from typing import Union, Dict
import pandas as pd
from PIL import Image
from pathlib import Path
from config import logger

class DatasetCreator(Dataset):
    def __init__(
            self, data_frame_pd: pd.DataFrame, 
            input_transform: Union[None, transforms.Compose, list] = None, 
            device: torch.device = torch.device("cpu"),
            dtype: torch.dtype = torch.float32):
        """
        The object created by this class contains the input(image) and target(label_grnd_truth) which can be feed to the neural model.
        In order to get i_th data of the object, use [i] after the object name,
        then it will return a dictionary whose keys are "image" and "label_grnd_truth", and its items are correponding values.

        Args:
            data_frame_pd (pd.DataFrame): The pandas data frame that has been created by one of the csv_files
            input_transform (Union[None, transforms.Compose, list], optional): 
                The transformation that should be used for pre-proccesing or augmentating input images.
                It also can be a list of transfomrations.
            device (torch.device, optional): based on it, the data will be stored on cpu or gpu.
            dtype (torch.dtype, optional): you can change the dtype to change required memory, but it also change the accuracy. 
        """
        super().__init__()
        # we need to shuffle data_frame to make the dataset more sutiable for training
        # we also need to reset the index after shuffling, otherwise df.loc[i] produces the same output as before suffling
        self.data_frame_pd = data_frame_pd.sample(frac=1).reset_index(drop=True)
        self.device = device
        self.dtype = dtype        
        
        if isinstance(input_transform, list):
            self.input_transform = transforms.Compose(input_transform)
        else:
            self.input_transform = input_transform

        self.to_tensor = transforms.ToTensor()
        # we need to calculate number of images for each label for imbalance data. 
        # so we can compensate it in the loss function by assigning different coefficients.
        self.num_images_per_label = self.calculate_number_of_images_for_each_label()


    def calculate_number_of_images_for_each_label(self) -> Dict[str, int]:
        """
        This function returns the number of images for each label in a dictionary format.
        We need to call this function to created a weighted loss for imbalanced dataset.

        Returns:
            Dict[str, int]: 
                Dict["valid"] is number of images for label valid.
                Dict["invalid"] is number of images for label invalid.
                Dict["unkown"] is number of images for label unkown.
                    we need to define unkown labes for making this class usable also for prediction.
        """
        num_valid = 0
        num_invalid = 0
        num_unkown = 0
        for i in range(len(self.data_frame_pd)):
            label = self.data_frame_pd.loc[i, "label"]
            assert (label in ["valid", "invalid", "unknown"]), logger.error(
            f"the corresponding label to index={i} from data_frame must be valid, invalid or unknown"
            )
            if label == "valid":
                num_valid = num_valid + 1
            elif label == "invalid":
                num_invalid = num_invalid + 1
            else:
                num_unkown = num_unkown + 1
        
        return {"valid": num_valid, "invalid": num_invalid, "unknown": num_unkown}
    
    def __len__(self) -> int:
        """
        This reserved function allows us to pass the object into the function "len".

        Returns:
            int: number of (image, label) in the dataset
        """
        return len(self.data_frame_pd)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        This reserved function allows us to get access of the data using the opeator []. 
        We can assging one output for the model since we have binary classification, but 
        I like more using softmax style, so I am going to assign two output in one-hot format for the model.
        And we need to be carefull and memorize that.

        Args:
            idx (int): the index of the data you want to get.

        Returns:
            Dict[str, torch.Tensor]:
                Dict["image"] is a pytorch tensor on the self.device with the shape of (3, H, W).
                Dict["label_grnd_truth"] is a pytorch tensor on the self.device with the shape of (2, ). 
        """

        label_name = self.data_frame_pd.loc[idx, "label"]
        # we are going two 
        # In the one-hot format, the model's target must be float, otherwise the loss_fn raises an error.
        if label_name == "valid":
            label = torch.tensor([1.0, 0.0], dtype=self.dtype)
        elif label_name == "invalid":
            label = torch.tensor([0.0, 1.0], dtype=self.dtype)
        else:
            label = torch.tensor([float("nan"), float("nan")], dtype=self.dtype)

        image_path = Path(self.data_frame_pd.loc[idx, "image_path"])
        image = Image.open(image_path).convert("RGB")
        # we need to tranform the image from PIL format to tensot format, 
        # otherwise it is not possible to store it on gpu.
        image = self.to_tensor(image).type(self.dtype)

        if self.input_transform:
            image = self.input_transform(image)
        return {
            "image": image.to(self.device), 
            "label_grnd_truth": label.to(self.device),
            "image_path": str(image_path)}

