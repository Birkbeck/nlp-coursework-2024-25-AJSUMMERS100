import nltk
import spacy
from pathlib import Path
import pandas as pd
from collections import Counter


nlp = spacy.load("en_core_web_sm")
nlp.max_length = 2000000

def part_a(path):
    pre_df = pd.read_csv(path)
    df = pd.DataFrame(pre_df)

    # Part I
    df.replace("Labour (Co-op)", "Labour", inplace = True)

    # Part II
    df = df[df['party'].notna()]
    df = df[df['party']!='Speaker']
    partys = Counter()
    for each in df['party']:
        partys[each] +=1
    top_four = Counter(partys).most_common(4)
    top_four_list=[]
    for i in range(4):
        top_four_list.append(top_four[i][0])
    for each in partys:
        if each not in top_four_list:
            df = df[df['party']!=each]
    new_counters = Counter()
    for each in df['party']:
        new_counters[each] +=1

    # Part III
    df = df[df['speech_class']=="Speech"]

    # Part IV
    df = df[df['speech'].str.len() >=1000]
    return df


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import numpy as np
def part_b(df):
    random_seed = 26
    np.random.seed(random_seed)
    tv = TfidfVectorizer(max_features = 3000)
    X = tv.fit_transform(df['speech'])
    y = df['party']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, stratify = y, random_state = random_seed)
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    path = Path.cwd() / "p2-texts" / "hansard40000.csv"
    
    # Part a
    df = part_a(path)
    print(df.shape)

    # Part b
    X_train, X_test, y_train, y_test = part_b(df)
    # Part c

    # Part d

    # Part e

