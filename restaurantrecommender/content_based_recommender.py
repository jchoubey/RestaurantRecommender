import pandas as pd
import numpy as np
from numpy.linalg import norm
from sklearn.pipeline import FeatureUnion
from transformers import *
from scipy.sparse import coo_matrix
import simplejson as json
from datetime import datetime
from sklearn.cross_validation import train_test_split
from restaurantrecommender.recommender import RestaurantRecommender


class ContentBasedRecommender(RestaurantRecommender):
    def __init__(self, user_review_business):
        """
          Creates an instance of Content Based Restaurant recommender.
          The init function also initializes the model params if not specified.
        """
        super().__init__(user_review_business)
        self.model = None
    
    
    def train_model(self):
        '''This function performs feature extraction on the yelp dataset 
        to prepare it for content based recommendation'''
        encoding_category = One_Hot_Encoder('categories', 'list', sparse=False)
        encoding_attribute = One_Hot_Encoder('attributes', 'dict', sparse=False)
        encoding_city= One_Hot_Encoder('city', 'value', sparse=False)
        rating = Column_Selector(['stars'])
        self.model = FeatureUnion([ ('cat', encoding_category),('attr', encoding_attribute),('city', encoding_city), ('rating', rating) ])
        self.model.fit(self.reviews_df)
        
        
    def create_user_profile(self, user_id):

        # Business for the review given by the user
        reviews_given_by_user = self.reviews_df.ix[self.reviews_df.user_id == user_id]
        reviews_given_by_user['stars'] = reviews_given_by_user['stars'] - float(self.reviews_df.average_stars[self.reviews_df.user_id == user_id])
        reviews_given_by_user = reviews_given_by_user.sort_values('business_id')

        # list of ids of the businesses reviewed by the user
        reviewed_business_id_list = reviews_given_by_user['business_id'].tolist()
        reviewed_business = self.reviews_df[self.reviews_df['business_id'].isin(reviewed_business_id_list)]
        reviewed_business = reviewed_business.sort_values('business_id')

        features = self.model.transform(reviewed_business)
        profile = np.matrix(reviews_given_by_user.stars) * features
    
        return profile


    def make_recommendation(self, user_id):
        
        user_profile = self.create_user_profile(user_id)
    
        test_frame = self.reviews_df[0:1000]
        test_frame = self.reviews_df.sort_values('business_id')
        business_id_list = self.reviews_df['business_id'].tolist()
        features = self.model.transform(test_frame)
        similarity = np.asarray(user_profile * features.T) * 1./(norm(user_profile) * norm(features, axis = 1))
        
        restaurant= []
        index_arr = (-similarity).argsort()[:10][0][0:10]
        for i in index_arr:
            restaurant.append(self.reviews_df[self.reviews_df.business_id == business_id_list[i]])
            
        return restaurant