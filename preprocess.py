import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from pyvi import ViPosTagger, ViTokenizer
from sentence_transformers import SentenceTransformer, util

def remove_emoji(text):
    """Removes emojis from the given text."""
    emoji_pattern = re.compile("["
                               u"\U0001F1E0-\U0001F1FF"  # emojis
                               u"\U00002700-\U000027BF"  # dingbats
                               u"\U0001f600-\U0001f64F"  # emoticons
                               u"\U0001f300-\U0001f5FF"  # symbols & pictographs
                               u"\U0001f680-\U0001f6FF"  # transport & object symbols
                               u"\U0001f1A0-\U0001f1FF"  # flags (iOS)
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


def find_noun_word(list_text):
    words = []
    for i in range(len(list_text)):
        list_text_array = list_text[i].split()
        # print(list_text_array)
        for j in range(len(list_text_array)):
            # print(ViPosTagger.postagging(list_text_array[j])[1])
            if ViPosTagger.postagging(list_text_array[j])[1][0].startswith('N'):
                words.append(list_text_array[j])
    return words





