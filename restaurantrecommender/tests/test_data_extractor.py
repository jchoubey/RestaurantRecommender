import unittest
import os
import pandas as pd
from restaurantrecommender.data_extractor import extract_data, process_data


class TestDataExtractor(unittest.TestCase):

    def test_extract_zip(self):
        """
            Checks that the downloaded archive file is extracted correctly.
        """
        # extract_data(tar_filename='yelp_dataset.tar', data_directory='../../data/raw')
        self.assertTrue(os.path.exists('../../data/raw/yelp_academic_dataset_business.json'))
        self.assertTrue(os.path.exists('../../data/raw/yelp_academic_dataset_user.json'))
        self.assertTrue(os.path.exists('../../data/raw/yelp_academic_dataset_review.json'))

    def test_process_data(self):
        """
            Checks if the function produces the merged clean file given input folder and filer
        """
        # process_data('../../data/raw', '../../data/clean', 'restaurant', 'philadelphia')
        self.assertTrue(os.path.exists("../../data/clean/user_business_review.csv"))

    def test_csv_structure(self):
        """
            Checks if the csv structure has the right number of columns.
        """
        user_business_review = pd.read_csv("../../data/clean/user_business_review.csv")
        self.assertTrue(len(user_business_review.columns), 5)

    def test_csv_column_names(self):
        user_business_review = pd.read_csv("../../data/clean/user_business_review.csv")
        self.assertTrue('business_id' in user_business_review.columns)
        self.assertTrue('user_id' in user_business_review.columns)
        self.assertTrue('stars' in user_business_review.columns)

