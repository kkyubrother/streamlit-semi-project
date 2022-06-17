"""
# My first app
Here's our first attempt at using data to create a table:
"""
import json
import streamlit as st
import numpy as np
import random
import string
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


with open("analyzed_data.json", encoding='utf-8') as f:
    analyzed_data = json.load(f)
df = pd.read_json("yes24.preprocessed.json")

tfidf_matrix = TfidfVectorizer().fit_transform(analyzed_data)
# print('TF-IDF 행렬의 크기(shape) :', tfidf_matrix.shape)

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
# print('코사인 유사도 연산 결과 :' , cosine_sim.shape)

title_to_index = dict(zip(df['title'][:100], df.index[:100]))


def get_recommendations(title, cosine_sim=cosine_sim):
    # 선택한 영화의 타이틀로부터 해당 영화의 인덱스를 받아온다.
    idx = title_to_index[title]

    # 해당 영화와 모든 영화와의 유사도를 가져온다.
    sim_scores = list(enumerate(cosine_sim[idx]))

    # 유사도에 따라 영화들을 정렬한다.
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # 가장 유사한 10개의 영화를 받아온다.
    sim_scores = sim_scores[1:11]

    # 가장 유사한 10개의 영화의 인덱스를 얻는다.
    movie_indices = [idx[0] for idx in sim_scores]

    # 가장 유사한 10개의 영화의 제목을 리턴한다.
    return df['title'].iloc[movie_indices], sim_scores


st.title("미리보기 기반 소설 추천 서비스")
novel_title = st.text_input("좋아하는 소설 제목을 입력하시오.")
st.write(f"검색 대상: {novel_title}")


# random_data = [
#     ["".join([random.choice(string.ascii_letters) for _ in range(random.randint(1, 16))]) for _ in range(16)],
#     np.random.rand(16)
# ]
# df = pd.DataFrame(zip(random_data[0], random_data[1]), columns=['name', 'acc'])
# df.columns = ['name', 'acc']
# df.sort_values(by=['acc'], ascending=False)
# print(df)

st.write("Here's our first attempt at using data to create a table:")
try:
    r = get_recommendations(novel_title)
    st.write(pd.DataFrame({"name": r[0], "score": [round(x[1], 3) * 100 for x in r[1]]}))
except KeyError:
    st.text("데이터가 없습니다.")
