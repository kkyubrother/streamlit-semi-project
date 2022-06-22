"""
# My first app
Here's our first attempt at using data to create a table:
"""
import json
import streamlit as st
import pandas as pd
from gensim.models import doc2vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@st.cache
def load_analyzed_data(without_sent: bool):
    """데이터 불러오기"""
    if without_sent:
        analyzed_data_path = "analyzed_data.without_sent.json"
        pre_processed_data_path = "yes24.preprocessed.without_sent.json"

    else:
        analyzed_data_path = "analyzed_data.with_sent.json"
        pre_processed_data_path = "yes24.preprocessed.with_sent.json"

    with open(analyzed_data_path, encoding='utf-8') as f:
        analyzed_data = json.load(f)
    df = pd.read_json(pre_processed_data_path)
    tfidf_matrix = TfidfVectorizer().fit_transform(analyzed_data)
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    title_to_index = dict(zip(df['title'], df.index))

    return title_to_index, cosine_sim, df


@st.cache
def load_doc2vec_data():
    return doc2vec.Doc2Vec.load("dart.doc2vec")


@st.cache
def get_recommendations(title, without_sent: bool):
    cosine_sim, title_to_index, df = load_analyzed_data(without_sent)

    # 선택한 제목에서 해당 책의 인덱스를 받아온다.
    idx = title_to_index[title]

    # 해당 책과 모든 책과의 유사도를 가져온다.
    sim_scores = list(enumerate(cosine_sim[idx]))

    # 유사도에 따라 책들을 정렬한다.
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # 가장 유사한 10개의 책을 받아온다.
    sim_scores = sim_scores[1:11]

    # 가장 유사한 10개의 책의 인덱스를 얻는다.
    movie_indices = [idx[0] for idx in sim_scores]

    # 가장 유사한 10개의 책의 제목을 리턴한다.
    return df['title'].iloc[movie_indices], sim_scores


def set_tfidf_column(col: st.delta_generator.DeltaGenerator, title: str, without_sent: bool):
    col.write("IF-IDF:")
    try:
        r = get_recommendations(title, without_sent)
        col.write(pd.DataFrame({"name": r[0], "score": [round(x[1], 3) * 100 for x in r[1]]}))
    except KeyError:
        col.text("데이터가 없습니다.")


def set_doc2vec_column(col: st.delta_generator.DeltaGenerator, title: str):
    model = load_doc2vec_data()
    col.write("Doc2Vec:")
    try:
        similar_doc = model.dv.most_similar(title)
        col.write(pd.DataFrame(similar_doc, columns=["제목", "유사도"]))
    except KeyError:
        col.text("데이터가 없습니다.")


def set_search_title_column(col, without_sent: bool):
    _, _, df = load_analyzed_data(without_sent)
    title = st.text_input("기억이 잘 나지 않는 책 입력하시오.")
    col.write(f"{title}")

    if title:
        col.write([x for x in df.title if title in x])
    else:
        col.write(df.title)


# 제목
st.title("미리보기 기반 소설 추천 서비스")

# 불용어 사용 선택
checkbox_btn = st.checkbox('불용 문장 처리')

novel_title = st.text_input("좋아하는 소설 제목을 입력하시오.")
st.write(f"검색 대상: {novel_title}")

col1, col2 = st.columns(2)
set_tfidf_column(col1, novel_title, checkbox_btn)
set_doc2vec_column(col2, novel_title)

set_search_title_column(st, novel_title)

