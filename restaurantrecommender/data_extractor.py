import pandas as pd
import tarfile
import os


def extract_data(tar_filename='yelp_dataset', data_directory='./data/raw/'):
    """
        This function extracts the data from the yelp zip file and writes the results to raw data folder.
    """
    if not os.path.exists(data_directory):
        os.makedirs(data_directory)
        
    print('extracting contents')
    file = tarfile.open(os.path.join(data_directory, tar_filename))
    file.extractall(path=data_directory)

    
def process_data(
        raw_data_directory='./data/raw',
        clean_data_directory='./data/clean',
        filter_category='restaurant',
        filter_city='philadelphia'):
    """
     This function filters the business and reviews data set and  writes the results to the data/clean folder.
     
    """

    if not os.path.exists(clean_data_directory):
        os.makedirs(clean_data_directory)

    business = pd.read_json(os.path.join(raw_data_directory, 'yelp_academic_dataset_business.json'), lines=True)
    business = business[['business_id', 'name', 'categories', 'attributes', 'city', 'stars']]
    business.rename({'name': 'business_name'}, axis=1, inplace=True)
    business.dropna(axis=0, how='any', inplace=True)
    str_cols = ['business_name', 'categories', 'attributes', 'city']
    business[str_cols] = business[str_cols].apply(func=lambda x: x.str.strip().str.lower(), axis=1)
    restaurant = business[(business['categories'].str.contains(filter_category))]
    restaurant.drop(axis=1, columns=['categories'], inplace=True)
    
    user = pd.read_json(os.path.join(raw_data_directory, 'yelp_academic_dataset_user.json'), lines=True)
    user = user[['user_id', 'name', 'average_stars']]
    user.rename({'name': 'user_name'}, axis=1, inplace=True)
    
    review = pd.read_json(os.path.join(raw_data_directory, 'yelp_academic_dataset_review.json'), lines=True)
    review = review[['user_id', 'business_id', 'stars']]
    
    user_review = pd.merge(left=user, right=review, how='inner', on='user_id')
    user_review_business = pd.merge(left=restaurant, right=user_review, how='inner', on='business_id')
    user_review_business = user_review_business[user_review_business['city'] == filter_city]
    user_review_business.to_csv(os.path.join(clean_data_directory, 'user_business_review.csv'), index=False)

    return user_review_business


def main():
    """
       This function executes the steps sequentially and writes the results to data folder.
    """
    extract_data()
    user_review_business = process_data()
    directory = './output'
    if not os.path.exists(directory):
        os.makedirs(directory)
    user_review_business.to_pickle("./output/user_review_business.pkl")

    
if __name__ == "__main__":
    main()