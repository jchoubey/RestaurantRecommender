import pandas as pd
import tarfile
import requests
import os

def extract_data(tar_filename='yelp_dataset.tar',
                  data_directory='../data/raw'):
    '''
        This function extracts the data from the yelp zip file and writes the results to raw data folder.
    '''
    if not os.path.exists(data_directory):
        os.makedirs(data_directory)
        
    print('extracting contents')
    file = tarfile.open(os.path.join(data_directory,tar_filename))
    file.extractall(path=data_directory)

def process_data(
        raw_data_directory='../data/raw',
        clean_data_directory='../data/clean',
        filter_category='restaurant',
        filter_city='portland'):
    '''
     This function cleans the raw data set and writes the results to the data/clean folder.
     
     restaurant = This is the data frame for restaurants
     review = This is the data frame that contains reviews for restuarants. 
     
     num_users = unique users from reviews
     num_restaurants = unique restaurants from dataframe. 
     
    '''
    
    if not os.path.exists(clean_data_directory):
        os.makedirs(clean_data_directory)
    
    business = pd.read_json(os.path.join(raw_data_directory, 'yelp_academic_dataset_business.json'), lines = True)
    business = business[['business_id','name','city','categories']]
    #drop na
    business.dropna(axis = 0, how = 'any', inplace = True)
    str_cols = ['name','city','categories']
    #lowercase string
    business[str_cols] = business[str_cols].apply(func = lambda x: x.str.strip().str.lower(), axis = 1)
    
    restaurant = business[(business['categories'].str.contains(filter_category)) & (business['city'] == filter_city)]
    restaurant.drop(axis = 1, columns = ['categories', 'city'], inplace = True)
    restaurant.to_csv(os.path.join(clean_data_directory, 'restaurant.csv'))
    num_restaurant = restaurant.business_id.unique().shape[0]
    
    review = pd.read_json(os.path.join(raw_data_directory, 'yelp_academic_dataset_review.json'), lines=True)
    review = review[['user_id', 'business_id', 'stars']]
    review = pd.merge(left = restaurant, right = review, how = 'inner', on = 'business_id')
    review.to_csv(os.path.join(clean_data_directory, 'review.csv'))

    num_users = review.user_id.unique().shape[0]
    return restaurant, review, num_restaurant, num_users

def main():
   '''
       This function executes the steps sequentially and writes the results to data folder.
   '''
   extract_data()
   restaurant, review, num_restaurant, num_users = process_data()
   directory = './output'
   if not os.path.exists(directory):
        os.makedirs(directory)
   restaurant.to_pickle("./output/restaurant.pkl")
   review.to_pickle("./output/review.pkl")
    
    
if __name__ == "__main__":
    main()