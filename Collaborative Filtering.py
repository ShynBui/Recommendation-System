import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from pyvi import ViPosTagger, ViTokenizer
from sentence_transformers import SentenceTransformer, util
import os
from sklearn.metrics.pairwise import cosine_similarity
from underthesea import word_tokenize, pos_tag, sent_tokenize
from surprise import *
from surprise.model_selection.validation import cross_validate
import pickle




if __name__ == "__main__":

    #load file rating
    product_rating = pd.read_csv(
        os.path.join(os.getcwd(),'data/Products_ThoiTrangNam_rating_raw.csv'),
        delimiter='\t')

    with open(os.path.join(os.getcwd(),"data/SVD_recommender.pkl"), "rb") as f:
        model = pickle.load(f)

    #Gia su chon user 127
    userId = 127
    df_score = product_rating.copy()[["product_id"]]

    df_score['EstimateScore'] = df_score['product_id'].apply(
        lambda x: model.predict(userId, x).est)  # est: get EstimateScore
    df_score = df_score.sort_values(by=['EstimateScore'], ascending=False)

    df_score = df_score.drop_duplicates()

    #Chon ratting >= 3

    print(df_score[df_score.EstimateScore >= 3])












