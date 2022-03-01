import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, save_npz
from pandas.api.types import CategoricalDtype
import pickle
import os
import implicit
import random


class ALSRestaurantRecommender:
    def __init__(self, restaurant_file_path, reviews_file_path, factors=20, regularization=0.1, iterations=50):
        """
          Creates an instance of ALS Restaurant recommender.

          The init function also initializes the model params if not specified.
        """
        self.restaurant_df = ALSRestaurantRecommender._load_csv_file(restaurant_file_path)
        self.reviews_df = ALSRestaurantRecommender._load_csv_file(reviews_file_path)
        self.iterations = iterations
        self.factors = factors
        self.regularization = regularization
        self.als_model = implicit.als.AlternatingLeastSquares(factors=self.factors, regularization=self.regularization,
                                                              iterations=self.iterations);
        self.sparse_restaurant_user = None
        self.sparse_user_restaurant = None

    @staticmethod
    def _load_csv_file(filepath):
        """
          This function loads a file into memory for computation
        """
        return pd.read_csv(filepath)

    def _create_sparse_matrix(self):
        """
           This function takes a dataframe and creates a sparse user-restaurant matrix and
           sparse restaurant-user matrix using csr matrix

           returns:
           sparse item-user matrix, user-item matrix
        """
        unique_users = list(self.reviews_df['user_id'].unique())
        unique_restaurant = list(self.reviews_df['business_id'].unique())
        rating = self.reviews_df['stars'].tolist()

        # converting users and restaurants into numerical ids
        rows = self.reviews_df.user_id.astype(CategoricalDtype(categories=unique_users)).cat.codes
        cols = self.reviews_df.business_id.astype(CategoricalDtype(categories=unique_restaurant)).cat.codes

        self.reviews_df['users_id_code'] = rows
        self.reviews_df['business_id_code'] = cols
        sparse_restaurant_user = csr_matrix((rating, (cols, rows)), shape=(len(unique_restaurant), len(unique_users)))
        sparse_user_restaurant = csr_matrix((rating, (rows, cols)), shape=(len(unique_users), len(unique_restaurant)))
        self.sparse_restaurant_user = sparse_restaurant_user
        self.sparse_user_restaurant = sparse_user_restaurant

    def get_sparsity_score(self):
        """
            This function prints the sparsity of review matrix.
        """
        sparsity_score = 1 - self.sparse_matrix.nnz / (sparse_matrix.shape[0] * sparse_matrix.shape[1])
        print("Sparsity Score = ", sparsity_score)
        return sparsity_score

    def _train_test_split(self, test_size):
        """
            This function takes a user-restaurant matrix and splits it into train/test data.
            We mask a percentage of ratings from training set.
        """
        self._create_sparse_matrix()
        ratings = self.sparse_user_restaurant
        test_set = ratings.copy()
        test_set[test_set != 0] = 1
        training_set = ratings.copy()
        user_restaurant_interaction = training_set.nonzero()
        interaction_index_pair = list(zip(user_restaurant_interaction[0], user_restaurant_interaction[1]))
        random.seed(0)
        test_set_size = int(np.ceil(test_size * len(interaction_index_pair)))
        test_samples = random.sample(interaction_index_pair, test_set_size)
        user_index = [index[0] for index in test_samples]
        restaurant_index = [index[1] for index in test_samples]
        training_set[user_index, restaurant_index] = 0
        training_set.eliminate_zeros()
        return training_set, test_set

    def user_rated_restaurants(self, user_id):
        """
            Given a user_id, returns a list of restaurants that are rated by the user.
            :param user_id
        """
        restaurant = [self.reviews_df.name.loc[self.reviews_df.users_id_code == user_id].iloc[0]]
        print("Rated By User", user_id)
        pd.DataFrame(restaurant)
        return restaurant

    def fit_model(self, test_size=0.2):
        """
          This function fits an ALS Model for the restaurant dataset which is used to get recommendation for a user.
          :param test_size the size for train test split
        """
        train_data, test_data = self._train_test_split(test_size)
        self.als_model.fit(train_data)

    def make_recommendation(self, user_id):
        """
            Given a user id this function recommends a list of restaurants that match user profile.
            :param user_id the id of the user
        """
        ids, scores = self.als_model.recommend(user_id, self.sparse_user_restaurant[user_id])
        restaurant = []
        for id in ids:
            restaurant.append(self.reviews_df.name.loc[self.reviews_df.business_id_code == id].iloc[0])
        print("Recommendation For User", user_id)
        print(pd.DataFrame(restaurant))

    def similar_restaurants(self, business_id, no_similar):
        """
            Given a business Id and No of results to return, this function returns a list of similar restaurants.
            :param business_id the id of the restaurant
            :param no_similar no of similar results to return
            :return restaurant list.
        """
        ids, scores = self.als_model.similar_items(business_id, no_similar)
        restaurant = []
        for id in ids:
            restaurant.append(self.reviews_df.name.loc[self.reviews_df.business_id_code == id].iloc[0])
        print("Restaurants Similar to", self.reviews_df.name.loc[self.reviews_df.business_id_code == id].iloc[0])
        print(pd.DataFrame(restaurant))
        return restaurant

    def save_pickle_model(self, als_pickle_file="./output/als.pickle"):
        """
          Saves current model to filepath specified by the user.
          :param als_pickle_file filepath specified by the yser.
          :return none
        """
        with open(als_pickle_file, "wb") as f:
            print(f"Saving ALS Model to {als_pickle_file}")
            pickle.dump(self.model, f)


def main():
    recommender = ALSRestaurantRecommender("../data/clean/restaurant.csv", "../data/clean/review.csv")
    recommender.fit_model()
    recommender.make_recommendation(12)
    recommender.similar_restaurants(1, 4)


if __name__ == "__main__":
    main()

