import unittest
import os
import pandas as pd
from restaurantrecommender.recommendationsystem import RecommendationSystem
import implicit


class TestRecommendationSystem(unittest.TestCase):

    rr = None

    @classmethod
    def setUpClass(cls):
        print("ran set up")
        cls.rr = RecommendationSystem("../../data/clean/user_business_review.csv", "ALSRecommender")

    @classmethod
    def tearDownClass(cls):
        cls.rr = None

    def test_constructors(self):
        """
            Tests whether the constructor initializes correctly.
        """
        self.assertTrue(self.rr.recommender is not None)

    def test_object_type(self):
        """
            Tests object type is ALS based on the input.
        """
        self.assertTrue(type(self.rr.recommender.model), implicit.als.AlternatingLeastSquares)

    def default_model_params(self):
        """
            Tests default model params.
        """
        self.assertEqual(self.rr.recommender.model.factors, 20)
        self.assertEqual(self.rr.recommender.model.regularization, 0.1)
        self.assertEqual(self.rr.recommender.model.factors, 50)

    def test_make_recommendation(self):
        """
            Tests whether make recommendation returns restaurants.
        """
        restaurant = self.rr.make_recommendation(100)
        self.assertEqual(len(restaurant), 10)

    def test_similar_restaurants(self):
        """
            Tests whether similar restaurant returns restaurants.
        """
        restaurant = self.rr.similar_restaurants(1, 5)
        self.assertEqual(len(restaurant), 5)

    def test_create_sparse_matrix(self):
        """
            Tests whether sparse matrix format is correctly returned or not.
        """
        self.assertEqual(self.rr.recommender.sparse_restaurant_user.getformat(), "csr")

    def test_create_sparse_matrix(self):
        """
            Tests whether sparse matrix is created correctly or not.
        """
        self.rr.recommender.create_sparse_matrix()
        self.assertEqual(self.rr.recommender.sparse_restaurant_user.get_shape(), (5857, 209582))