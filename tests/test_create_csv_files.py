
import unittest
import pandas as pd
import math
from pathlib import Path
from torch.utils.data import DataLoader
from textile_classification.data_setters import create_csv_files_for_train_val_test_phases
from config import TESTS_DIR, DATA_DIR

class TestCreateCSVFiles(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls) -> None:
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
        
        csv_train = csv_dir.joinpath("train.csv")
        csv_val = csv_dir.joinpath("validation.csv")
        csv_test = csv_dir.joinpath("test.csv")
        cls.df_train = pd.read_csv(csv_train)
        cls.df_val = pd.read_csv(csv_val)
        cls.df_test = pd.read_csv(csv_test)
    
    def test_csv_files_have_no_common(self):
        """
        train, val and test data must not have any common image
        """
        images_train_list = [self.df_train.loc[i, "image_path"] for i in range(len(self.df_train))]
        images_val_list = [self.df_val.loc[i, "image_path"] for i in range(len(self.df_val))]
        images_test_list = [self.df_test.loc[i, "image_path"] for i in range(len(self.df_test))]

        images_train_set = set(images_train_list)
        images_val_set = set(images_val_list)
        images_test_set = set(images_test_list)

        self.assertFalse(images_train_set.intersection(images_val_set))
        self.assertFalse(images_train_set.intersection(images_test_set))
        self.assertFalse(images_val_set.intersection(images_test_set))



    def test_all_data_are_included_in_csv_files(self):
        """
        make sure no image has been left, and
        data has bee splitted into train, val and test correcly based on the portions. 
        """
        for label in ["valid", "invalid"]:
            full_dir = DATA_DIR.joinpath(label)
            image_paths = []
            for each_extension in self.extensions:
                image_paths.extend([x for x in Path(full_dir).glob(f"*.{each_extension}")])
        
            num_images_all = len(image_paths)
            filt = self.df_train["label"] == label
            num_images_train = len(self.df_train[filt])
            
            filt = self.df_val["label"] == label
            num_images_val = len(self.df_val[filt])
            
            filt = self.df_test["label"] == label
            num_images_test = len(self.df_test[filt])

            self.assertEqual(num_images_all, num_images_train + num_images_val + num_images_test)
            self.assertEqual(num_images_train, math.floor(num_images_all * self.portion_train))
            self.assertEqual(num_images_val, 
                             (math.floor(num_images_all * (self.portion_val + self.portion_train)) - 
                              math.floor(num_images_all * self.portion_train)))
        

if __name__ == '__main__':
    unittest.main()

