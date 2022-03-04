import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, save_npz
from pandas.api.types import CategoricalDtype
import pickle
import os
import implicit
import random

# local modules
from restaurantrecommender.alsrecommender import AlsRecommender
from restaurantrecommender.constants import RecommenderType
from restaurantrecommender.exception_errors import InvalidRestaurantRecommenderType


class RecommendationSystem:
    def __init__(self, user_review_business, recommender_type):
        if RecommenderType(recommender_type) is RecommenderType.AlsRecommender:
            self.recommender = AlsRecommender(user_review_business)
        else:
            raise InvalidRestaurantRecommenderType("recommender type not supported")
        self._train_model()

    def _train_model(self):
        """
          This function fits a Model for the restaurant dataset which is used to get recommendation for a user.
          :param test_size the size for train test split
        """
        self.recommender.train_model()

    def make_recommendation(self, user_id):
        """
            Given a user id this function recommends a list of restaurants that match user profile.
            :param user_id the id of the user
        """
        self.recommender.make_recommendation(user_id)

    def similar_restaurants(self, business_id, no_similar):
        """
            Given a business Id and No of results to return, this function returns a list of similar restaurants.
            :param business_id the id of the restaurant
            :param no_similar no of similar results to return
            :return restaurant list.
        """
        self.recommender.similar_restaurants(business_id, no_similar)

    def save_pickle_model(self, file_path):
        """
          Saves current model to filepath specified by the user.
          :return none
        """
        self.recommender.save_pickle_model(file_path)


def main():
    recommendationsystem = RecommendationSystem("../data/clean/user_business_review.csv", "ALSRecommender")
    recommendationsystem.make_recommendation(100)
    recommendationsystem.similar_restaurants(1, 10)
    recommendationsystem.save_pickle_model()


if __name__ == "__main__":
    main()
