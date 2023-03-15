
import unittest
import pandas as pd
from torch.utils.data import DataLoader
from textile_classification.data_setters import (DatasetCreator, 
                                                 create_csv_files_for_train_val_test_phases)

from textile_classification.utils.data_set_visualization import plot_images
from textile_classification.utils.metric_calculation_visualization import change_labels_from_one_hot_fromat_to_binary_format
from config import TESTS_DIR

class TestDataVisualization(unittest.TestCase):
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

    def test_plot_images(self):
        """
        This function plot and save a bunch of images with its ground truth labels.
        Take look at test/artifacts/test_data_visualization.png to see what it has plotted.
        """
        batch = next(iter(self.dl))
        labels_grnd_truth = change_labels_from_one_hot_fromat_to_binary_format(batch["label_grnd_truth"])
        fig = plot_images(images=batch["image"], 
                          labels_grnd_truth=labels_grnd_truth)
        
        fig_file = TESTS_DIR.joinpath("artifacts", "test_data_visualization.png")
        fig.savefig(fig_file)

if __name__ == '__main__':
    unittest.main()