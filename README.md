## Data 515: Software Design for Data Science

## Restaurant Recommendation System
Authors: Poornima Muthukumar, Juhi Choubey

## Introduction
Recommendation Systems return the most relevant and accurate results (products, restaurants, books, travel plans, movies, tv-shows) to the user by filtering useful results from a huge pool of information. Recommendation systems discover data patterns in the data set by learning about customers’ choices and produce the outcome that correlates to their interest or preference. 

As a developer implementing a recommendation system can be a daunting task. With this project, we simplify integrating a recommendation system for recommending restaurants to users. Businesses can benefit from a good recommendation system as it increases user engagement and creates stickiness on the platform. In this project we build a python package that provides developers with the ability to easily integrate two kinds of recommendation systems - one is the collaborative filtering model and the other is a content based filtering model. 

## Goal

The goal of the project is to use the yelp dataset to build a restaurant recommendation system that recommends restaurants to users based on two different approaches 

The collaborative filtering recommendation model works by searching a large group of users and finding a smaller set of users with tastes similar to the particular user. It looks at the restaurant they like and creates a ranked list of suggested restaurants. 
The content based filtering recommendation model works by recommending restaurants to users based on similar restaurant categories and dominant topic keyworks, thus suggesting restaurants that alight with a user’s preferences. 

## Dataset Decription

## Data Processing

## Project Folder Structure

## Installaton and user guide on how to use RestaurantRecommender
To install and use RestaurantRecommender, you can follow the below steps or refer to the example section below.

1. Clone the repository:
	```git clone https://github.com/Poornima-Muthukumar/RestaurantRecommender```
2. 
	```cd RestaurantRecommender```
3. 
	```python setup.py install```
  
## Example

   ```
   from restaurantrecommender import RecommendationSystem
   
   r = RecommendationSystem("../data/clean/user_business_review.csv", "CollaborativeBasedRecommender")
   
   ```
   To get the top 10 recommendation for the user you can input
   ```
   user_id = 100
   restaurant = r.make_recommendation(user_id)
   ```
   You can also input an integer to specify the number of recommendations. The example below outputs the top 10 recommendatios that are similar to restaurant_id = 3
   ```
   top = 10
   restaurant_id = 3
   restaurant = r.similar_restaurants(restaurant_id, top)
   ```

## Limitations

Currently, we only support two kids of recommendation - Collaborative and Content Based Recommendation. The collaborative based approach is based on ALS Algorithm. Later we can expand our approach to support other recommendation types as well. 
The package does not ahve an UI interface in its current state, but providing one in the future would make the interaction easy for non-technical users.

