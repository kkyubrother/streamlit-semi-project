"""
# My first app
Here's our first attempt at using data to create a table:
"""
import json
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


st.title("미리보기 기반 소설 추천 서비스")


checkbox_btn = st.checkbox('불용 문장 처리')
if checkbox_btn:
    with open("analyzed_data.without_sent.json", encoding='utf-8') as f:
        analyzed_data = json.load(f)
    df = pd.read_json("yes24.preprocessed.without_sent.json")

else:
    with open("analyzed_data.with_sent.json", encoding='utf-8') as f:
        analyzed_data = json.load(f)
    df = pd.read_json("yes24.preprocessed.with_sent.json")

tfidf_matrix = TfidfVectorizer().fit_transform(analyzed_data)
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
title_to_index = dict(zip(df['title'], df.index))


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


novel_title = st.text_input("기억이 잘 나지 않는 책 입력하시오.")
st.write(f"{novel_title}")

if novel_title:
    st.write([x for x in df.title if novel_title in x])
else:
    st.write(df.title)


novel_title = st.text_input("좋아하는 소설 제목을 입력하시오.")
st.write(f"검색 대상: {novel_title}")
if novel_title.startswith('"') and novel_title.endswith('"'):
    novel_title = novel_title[1:-2]

st.write("Here's our first attempt at using data to create a table:")
try:
    r = get_recommendations(novel_title)
    st.write(pd.DataFrame({"name": r[0], "score": [round(x[1], 3) * 100 for x in r[1]]}))
except KeyError:
    st.text("데이터가 없습니다.")
