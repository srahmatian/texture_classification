
import unittest
import torch
from torch import nn
from textile_classification.models import PretrainedDenseNet

class TestPretrainedDenseNet(unittest.TestCase):

    def test_change_fc_layer(self):
        """
        In the function "change_fc_layer", 
        we change the only fc layer using self.model.classifier = nn.Linear(), so
        we need to check if there is only one fc layer in the pretrained model, and
        to check that if its name is "classifier" for having access to it. 
        """
        model = PretrainedDenseNet()
        fc_layers = []
        for module in model.model.modules():
            if isinstance(module, nn.Linear):
                fc_layers.append(module)
        
        self.assertEqual(len(fc_layers), 1)

        for name, child in model.model.named_children():
            if isinstance(child, nn.Linear):
                fc_layer_name = name
                break
        
        self.assertEqual(fc_layer_name, "classifier")
    
    def test_find_num_feaures_extraced_by_cnn_layers(self):
        """
        The original pre-trained model has 1024 features as the input to its FC layer.
        We nee to check if we can get the same number using find_num_feaures_extraced_by_cnn_layers.
        """
        model = PretrainedDenseNet()
        num_features_grnd_truth = model.model.classifier.in_features
        # I found that it dosn't matter what is the input size.
        # The image will always resize to (256, 256)
        x = torch.randn([1, 3, 256, 256])
        num_features_calculated = model.find_num_feaures_extraced_by_cnn_layers(x)

        self.assertEqual(num_features_grnd_truth, num_features_calculated)
    
    def test_number_of_outputs(self):
        """
        When you want to create the model you can assign the number of its ouput to any value, but
        In our classification task, we have two outputs in one-hot format for the model.
        With this function we can check the number of model's output
        """
        num_out_features = 2
        model = PretrainedDenseNet()
        # our images are in the shape of (3, 256, 256) before batching them.
        batch_size = 4
        x = torch.randn([batch_size, 3, 256, 256])
        num_last_hidden_features = model.find_num_feaures_extraced_by_cnn_layers(x)
        model = PretrainedDenseNet(new_in_fearues_of_fc=num_last_hidden_features, 
                                   new_out_features_of_fc=num_out_features)
        y = model(x)
        self.assertEqual(y.shape, torch.Size([batch_size, num_out_features]))



if __name__ == '__main__':
    unittest.main()