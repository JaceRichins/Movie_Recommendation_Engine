import pandas as pd
import numpy as np
import random
import warnings
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity


movies_description_df = pd.read_csv("movies_description.csv")
movies_df = pd.read_csv('movies.csv')

print(movies_description_df.columns)
print(movies_df.columns)