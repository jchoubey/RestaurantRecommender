import pandas as pd
import numpy as np
from pandas.api.types import CategoricalDtype
from scipy.sparse import csr_matrix, save_npz
import os


def create_sparse_matrix(data):
    """
       This function takes a dataframe and creates a sparse user-restaurant matrix and 
       sparse restaurant-user matrix using csr matrix
       
       returns:
       sparse item-user matrix, user-item matrix
    """
    # unique users
    users = list(np.sort(data.user_id.unique()))  
    
    # unique restaurants
    restaurant = list(np.sort(data.business_id.unique()))
    
    # rating
    rating = list(data.stars) 
    rows = data.user_id.astype(CategoricalDtype(categories=users)).cat.codes
    cols = data.business_id.astype(CategoricalDtype(categories=restaurant)).cat.codes
    
    sparse_restaurant_user = csr_matrix((rating, (cols, rows)), shape=(len(restaurant), len(users)))
    sparse_user_restaurant = csr_matrix((rating, (rows, cols)), shape=(len(users), len(restaurant)))
    print("sparse restaurant_user", sparse_restaurant_user.shape)
    print("sparse user_restaurant", sparse_user_restaurant.shape)
    return sparse_restaurant_user, sparse_user_restaurant


def main():
    data = pd.read_pickle("./output/review.pkl")
    sparse_restaurant_user, sparse_user_restaurant = create_sparse_matrix(data)
    directory = './output'
    if not os.path.exists(directory):
        os.makedirs(directory)
    save_npz("./output/sparse_restaurant_user.npz", sparse_restaurant_user)
    save_npz("./output/sparse_user_restaurant.npz", sparse_user_restaurant)


if __name__ == "__main__":
    main()