import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import argparse

parser = argparse.ArgumentParser(description='train for SLG')
parser.add_argument('--dataset', default="phoenix2014", help='the target dataset')
args = parser.parse_args()

target = args.dataset # phoenix2014, phoenix2014T, CSLDaily

df = pd.DataFrame(columns=['sentence', 'word_index', 'sentence_index', 'l1_label', 'l2_label'])

with open('./generation/description_{}.txt'.format(target), 'r', encoding='utf-8') as input_file:
    descriptions = input_file.readlines()

    cnt = -1
    for sen in descriptions:
        sen = sen.strip()
        if sen.startswith("1"):
            cnt+=1
        if sen.strip() == "" or sen.startswith("T"):
            continue
        new_row = {'sentence': sen[3:], 'word_index': cnt, 'sentence_index': sen[0]}
        df = df.append(new_row, ignore_index=True)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['sentence'])

l1 = 100 # the number of l1 nodes
kmeans = KMeans(n_clusters=l1, random_state=42)
kmeans.fit(X)

df.loc[df.index, 'l1_label'] = kmeans.labels_

for i in range(l1):
    print('process node' + str(i) + ' in l1')
    subset_df = df[df['l1_label'] == i]

    X = vectorizer.fit_transform(subset_df['sentence'])

    l2 = subset_df.shape[0] // 10 + 1
    kmeans = KMeans(n_clusters=l2, random_state=42)
    kmeans.fit(X)

    subset_df.loc[subset_df.index, 'l2_label'] = kmeans.labels_
    df.loc[subset_df.index, 'l2_label'] = subset_df['l2_label'].values

df.to_csv("description_{}.csv".format(target))
