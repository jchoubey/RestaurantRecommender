import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, save_npz
from pandas.api.types import CategoricalDtype
import pickle
import os
import implicit
import random
from constants import RecommenderType
from exception_errors import InvalidRestaurantRecommenderType


class RestaurantRecommender:
    def __init__(self, restaurant_file_path, reviews_file_path):
        self.restaurant_df = RestaurantRecommender._load_csv_file(restaurant_file_path)
        self.reviews_df = RestaurantRecommender._load_csv_file(reviews_file_path)

    @staticmethod
    def _load_csv_file(filepath):
        """
          This function loads a file into memory for computation
        """
        return pd.read_csv(filepath)

    def user_rated_restaurants(self, user_id):
        """
            Given a user_id, returns a list of restaurants that are rated by the user.
            :param user_id
        """
        restaurant = [self.reviews_df.name.loc[self.reviews_df.users_id_code == user_id].iloc[0]]
        print("Rated By User", user_id)
        pd.DataFrame(restaurant)
        return restaurant

    def save_pickle_model(self, model_pickle_file="./output/model.pickle"):
        """
          Saves current model to filepath specified by the user.
          :param model_pickle_file filepath specified by the yser.
          :return none
        """
        with open(model_pickle_file, "wb") as f:
            print(f"Saving Recommendation Model to {model_pickle_file}")
            pickle.dump(self.model, f)
