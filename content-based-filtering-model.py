import os
import pandas as pd
from pandas import DataFrame
import numpy as np
from numpy import dot
from numpy.linalg import norm

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing

# Import data
houses = pd.read_csv('./houses.csv')

# Format data

#BEDS, BATHS, SQ_FT, FLOORS, LAT, LONG (number)

houses_a = houses[['BEDS', 'BATHS', 'SQ_FT', 'FLOORS', 'LAT', 'LONG']]
house_data_c = []
for i in range(2985):
    temp = []
    for j in range(6):
        temp.append(list(houses_a.iloc[i])[j])
    house_data_c.append(temp)

#NEIGHBORHOOD, STYLE (words)
houses_n = houses[['NEIGHBORHOOD']]
houses_s = houses[['STYLE']]

neighbors = []
for i in range(2985):
    neighbors.append(list(houses_n.iloc[i])[0])
styles = []
for i in range(2985):
    styles.append(list(houses_s.iloc[i])[0])

count_vectorizer = CountVectorizer()
X = count_vectorizer.fit_transform(neighbors)
Y = count_vectorizer.fit_transform(styles)

encode_results = X.toarray()
encode_results_style = Y.toarray()

words = pd.DataFrame(encode_results/69)
words_style = pd.DataFrame(encode_results_style/13)

# Combine

for i in range(2985):
    house_data_c[i] = house_data_c[i] + \
        list(words.iloc[i])+list(words_style.iloc[i])

house_df = pd.DataFrame(house_data_c)
min_max_scaler = preprocessing.MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(house_df)
df_normalized = pd.DataFrame(np_scaled)
ls = df_normalized.values.tolist()
df_normalized.head(10)


def data_select_nor(id):
    index_number = list(houses['MLSA_ID']).index(id)
    house_select = ls[index_number]
    return house_select


def cosine_similarity(v1, v2):
    dot_x1_x2 = np.dot(v1, v2)
    x1 = np.array(v1)
    x2 = np.array(v2)
    n1_n = np.linalg.norm(x1)
    n2_n = np.linalg.norm(x2)
    result = dot_x1_x2/(n1_n*n2_n)
    return result


def recommend_list_nor(target):
    similarities = []
    for i in range(500):
        item = []
        item.append(i)
        item.append(cosine_similarity(target, ls[i]))
        similarities.append(item)
    return similarities


def recommend_top10_2_1_2(id_array):
    frames = []
    for id in id_array:
        target = data_select_nor(id)
        result = recommend_list_nor(target)
        result_list = DataFrame.from_records(result)
        result_list = result_list.sort_values(by=[1], ascending=False)
        top10 = result_list.head(10)
        frames.append(top10)
    total = pd.concat(frames)
    total = total.sort_values(by=[1], ascending=False)
    total_list = total.values
    for i in range(10):
        for j in range(i):
            if total_list[i][0] == total_list[j][0]:
                total_list = np.delete(total_list, i, 0)
                i = i - 1
                break
    res = DataFrame.from_records(total_list).head(10)
    print(res)
    top10_index_array = list(res[0])
    df = houses.loc[top10_index_array, :]
    return df


recommend_top10_2_1_2(['30634468', '30633486', '30631768'])
