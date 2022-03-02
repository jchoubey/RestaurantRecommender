import pandas as pd
import tarfile
import os


def extract_data(tar_filename='yelp_dataset.tar', data_directory='../data/raw'):
    """
        This function extracts the data from the yelp zip file and writes the results to raw data folder.
    """
    if not os.path.exists(data_directory):
        os.makedirs(data_directory)
        
    print('extracting contents')
    file = tarfile.open(os.path.join(data_directory,tar_filename))
    file.extractall(path=data_directory)


def process_data(
        raw_data_directory='../data/raw',
        clean_data_directory='../data/clean',
        filter_category='restaurant',
        filter_city='philadelphia'):
    """
     This function filters the business and reviews data set and  writes the results to the data/clean folder.
     
     restaurant = This is the data frame for restaurants
     data = This is the data frame that contains reviews for restaurant.
    """

    if not os.path.exists(clean_data_directory):
        os.makedirs(clean_data_directory)

    business = pd.read_json(os.path.join(raw_data_directory, 'yelp_academic_dataset_business.json'), lines=True)
    business = business[['business_id', 'name', 'city', 'categories']]
    business.rename({'name': 'business_name'}, axis=1, inplace=True)

    # drop na
    business.dropna(axis=0, how='any', inplace=True)
    str_cols = ['categories', 'city']

    # lowercase string
    business[str_cols] = business[str_cols].apply(func=lambda x: x.str.strip().str.lower(), axis=1)

    restaurant = business[(business['categories'].str.contains(filter_category))]
    restaurant.drop(axis=1, columns=['categories'], inplace=True)

    num_restaurant = restaurant.business_id.unique().shape[0]
    print(f"Number of Restaurant {num_restaurant}")

    review = pd.read_json(os.path.join(raw_data_directory, 'yelp_academic_dataset_review.json'), lines=True)
    review = review[['user_id', 'business_id', 'stars']]

    user = pd.read_json(os.path.join(raw_data_directory, 'yelp_academic_dataset_user.json'), lines=True)
    user = user[['user_id', 'name']]
    user.rename({'name': 'user_name'}, axis=1, inplace=True)

    user_review = pd.merge(left=user, right=review, how='inner', on='user_id')

    user_review_business = pd.merge(left=restaurant, right=user_review, how='inner', on='business_id')
    user_review_business = user_review_business[user_review_business['city'] == filter_city]

    user_review_business.drop(axis=1, columns=['city'], inplace=True)
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