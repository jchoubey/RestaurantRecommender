class InvalidRestaurantRecommenderType(Exception):
    """
        The exception error to throw when the requested recommender type is not supported
    """
    def __init__(self, message="Invalid Recommender Type Requested"):
        super().__init__(message)
