

import torch
from torch import nn
from typing import Union
from config import HUB_DIR_PRETRAINED, logger

class PretrainedDenseNet(nn.Module):
    """
    This model is one of the famous model in image classification. 
    And it is possilbe to use such a pretrained model from torchvision.models.
    the checkpoint of pretrained model already donwloaded from https://download.pytorch.org/models/densenet121-a639ec97.pth
    and it must be put in this directory: torch.hub.get_dir()/checkpoints
    But this model was trained on another kind of classification with differen number of labels, so
    we need to change its FC layer and fine tune the whole model.
    You can find more details about it on https://arxiv.org/abs/1608.06993
    
    """
    def __init__(self, new_in_fearues_of_fc: Union[int, None] = None, 
                 new_out_features_of_fc: Union[int, None] = None) -> None:
        """
        This models can not be used for different image sizes since
        the number of feature extarcted by cnn layers will change dependently.
        You also would need to use this model for different number of labels (out_feaures), so
        You would need to change the architecture of the FC layers based on your input image size and number of labels.
        For doing that you need to inspect the original architecture to find layers' name, then
        you will know how to access to them for chaning or replacing.

        Args:
            new_in_fearues_of_fc (Union[int, None], optional): If you pass None, it will not change.
            new_out_features_of_fc (Union[int, None], optional): If you pass None, it will not change.
        """
        super().__init__()

        self.model = self.get_original_model()
        if (new_in_fearues_of_fc is not None) or (new_out_features_of_fc is not None):
            self.change_fc_layer(new_in_fearues_of_fc, new_out_features_of_fc)
    
    def get_original_model(self) -> nn.Module:
        """
        This function take it look at HUB_DIR_PRETRAINED, 
        and create the densenet121 model by load its pth file in that directory. 

        Returns:
            nn.Module: the original pre-trained model
        """
        
        torch.hub.set_dir(HUB_DIR_PRETRAINED)
        original_model = torch.hub.load(HUB_DIR_PRETRAINED, "densenet121", 
                                        source='local', weights="DenseNet121_Weights.DEFAULT")
        
        # Make sure to set gradients of all model parameters to zero
        original_model.zero_grad()
        # Make all the layers unfrozen for fine tunning
        original_model.requires_grad_(requires_grad=True)
        return original_model
    
    def change_fc_layer(self, new_in_featues: int, new_out_features: int) -> None:
        """
        There is only one fully connected layer inside original model. 
        It is a Sequential layer named classifier.
        This function replace the classifier of original model to make it compatibel for new input and output sizes.
        we just change the linear layer. 
        we keep the dropout layer is its original version.

        Args:
            new_in_featues (int): num of input features for the FC layer
            new_out_features (int):number of classes (labels)
        """
        new_classifier = nn.Linear(in_features=new_in_featues, out_features=new_out_features)
        self.model.classifier = new_classifier

    def find_num_feaures_extraced_by_cnn_layers(self, x: torch.Tensor) -> int:
        """
        by passing your input image to this function,
        you will find the number of features extracted by whole CNN layers before the fully connedted layer.
        The output of this function can be used for change the FC layer.

        Args:
            x (torch.Tensor): is a tensor in shape of (batch_size, 3, height, width)

        Returns:
            int: the number of feautres extracted by whole CNN layers
        """

        assert (x.dim() == 4), logger.error(
            "the input of find_num_feaures_extraced_by_cnn_layers must have 4 dimenstions")
        
        model_removed_fc = nn.Sequential(*list(self.model.children())[:-1])
        # note that index 0 retrun batch_size
        num_feaures = model_removed_fc(x).shape[1]
        return num_feaures

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        This function determines how the model produce the output.
        Args:
            x (torch.Tensor): is a tensor in shape of (batch_size, 3, height, width)

        Returns:
            torch.Tensor: is a tensor in shape of (batch_size, num_labels) which is (batch_size, 2) in our task
        """
        return self.model(x)

