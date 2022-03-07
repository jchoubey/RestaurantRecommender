## Data 515: Software Design for Data Science

## Restaurant Recommendation System
Authors: Poornima Muthukumar, Juhi Choubey

## Introduction

## Objective

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

