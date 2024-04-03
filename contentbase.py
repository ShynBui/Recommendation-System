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


model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')

def get_policy():
    print("Mời bạn chon:\n1.Gợi ý sản phẩm có sẵn\n2.Nhập một phần gợi ý")
    choice = int(input())
    return choice

def process1(df, nhap, k = 3):  #Gợi ý sản phẩm có sẵn
    result = cosine_similarity(
        [df.loc[nhap, 'products_embedding']],
        df['products_embedding'].tolist()
    )
    sort_list = pd.DataFrame(result[0]).sort_values(by = 0, ascending = False)
    similar_items = sort_list.index.tolist()
    similar_items = similar_items[1: k + 1]
    return similar_items

def process2(df, nhap, k = 3):  #Nhập một phần gợi ý
    result = cosine_similarity(
        [model.encode(nhap)],
        df['products_embedding'].tolist()
    )
    sort_list = pd.DataFrame(result[0]).sort_values(by = 0, ascending = False)
    similar_items = sort_list.index.tolist()
    similar_items = similar_items[0: k]
    return similar_items

def show_result_policy(df, policy):
    print("Chúng tôi có những sản phẩm sau: ")
    for i, row in df.iterrows():
        print('Sản phẩm:', i, '\nThông tin sản phẩm', row['products'])
    if policy == 1:
        print("Bạn chọn: Gợi ý sản phẩm có sẵn")
        nhap = int(input('Mời bạn chọn sản phẩm: '))
        print('Bạn chọn: ', df.loc[nhap, 'products'])
    else:
        print("Bạn chọn: Nhập một phần gợi ý")
        nhap = input('Mời bạn nhập phần gợi ý: ')
        print('Bạn chọn: ', nhap)

    return nhap

def print_by_index(df, index):
    print('Sản phẩm:', index)
    for i in index:
        print(df.loc[i, 'products'])

def main():
    policy = get_policy()
    print('=' * 100)
    nhap = show_result_policy(df, policy)
    print('=' * 100)
    if policy == 1:
        result = process1(df, nhap)
        print('Top 3 sản phẩm liên quan là: ')
        print_by_index(df, result)
    else:
        result = process2(df, nhap)
        print('Top 3 sản phẩm liên quan là: ')
        print_by_index(df, result)
    print('=' * 100)


if __name__ == "__main__":

    with open(os.path.join(os.getcwd(),'data/data.txt'), 'r', encoding='utf-8') as file:
        # Đọc từng dòng của tập tin và lưu vào danh sách lines
        data = file.readlines()

    df = pd.DataFrame(data, columns =['products'])

    df["products_wt"] = df["products"].apply(lambda x: word_tokenize(x, format="text"))

    df['products_embedding'] = df['products_wt'].apply(lambda x: model.encode(x))

    main()









