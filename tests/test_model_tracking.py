
import unittest
import torch 
from torch.utils.data import DataLoader
import pandas as pd
from textile_classification.utils import (log_after_feeding_each_batch,  
                                          change_labels_from_one_hot_fromat_to_binary_format, 
                                          change_labels_from_probablity_format_to_deterministic_format,
                                          plot_images)
from textile_classification.data_setters import (create_csv_files_for_train_val_test_phases, 
                                                 DatasetCreator)
from config import TESTS_DIR

class TestModelTracking(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """set up resources that are needed for the entire test case"""
        csv_dir = TESTS_DIR.joinpath("artifacts")
        csv_dir.mkdir(parents=True, exist_ok=True)
        cls.extensions = ["png", "jpg", "jpeg", "jpe", "jfif"]
        cls.portion_train = 0.8
        cls.portion_val = 0.15
        
        create_csv_files_for_train_val_test_phases(
            extensions=cls.extensions, 
            portion_train=cls.portion_train, 
            portion_val=cls.portion_val, 
            csv_dir=csv_dir)
        
        csv_file = csv_dir.joinpath("validation.csv")
        # create data_frame, data_set, and data_loader
        df = pd.read_csv(csv_file)
        ds = DatasetCreator(df)
        batch_size = 32
        cls.dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    def test_log_after_feeding_each_batch(self):
        """
        This function compares predicted labels with ture labels, and 
        plots and saves the images wronlgy classified.
        Take look at test/artifacts/test_model_tracking_log_after_feeding_each_batch.png to see what it has plotted.
        """
        batch = next(iter(self.dl))
        labels_grnd_truth = batch["label_grnd_truth"]
        images = batch["image"]
        labels_predicted = torch.zeros_like(labels_grnd_truth)
        rand_nums = torch.rand(labels_predicted.shape[0], dtype=labels_predicted.dtype)
        labels_predicted[:, 0] = rand_nums
        labels_predicted[:, 1] = 1 - rand_nums

        labels_grnd_truth = change_labels_from_one_hot_fromat_to_binary_format(labels_grnd_truth)
        labels_predicted = change_labels_from_one_hot_fromat_to_binary_format(labels_predicted)
        labels_predicted = change_labels_from_probablity_format_to_deterministic_format(labels_predicted, 
                                                                                        is_one_hot_format=False)

        mask = labels_predicted != labels_grnd_truth
        if torch.any(mask) == True:
            fig = plot_images(images=(images[mask][0:16]).to("cpu"), 
                              labels_grnd_truth=labels_grnd_truth[mask][0:16], 
                              labels_predicted=labels_predicted[mask][0:16])
        
            fig_file = TESTS_DIR.joinpath("artifacts", "test_model_tracking_log_after_feeding_each_batch.png")
            fig.savefig(fig_file)

if __name__ == '__main__':
    unittest.main()