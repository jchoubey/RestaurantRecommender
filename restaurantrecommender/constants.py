import enum


class RecommenderType(enum.Enum):
    CollaborativeBasedRecommender = "CollaborativeBasedRecommender"
    ContentBasedRecommender = "ContentBasedRecommender"
