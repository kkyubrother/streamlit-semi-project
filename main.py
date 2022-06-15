"""
# My first app
Here's our first attempt at using data to create a table:
"""

import streamlit as st
import numpy as np
import random
import string
import pandas as pd

st.title("미리보기 기반 소설 추천 서비스")
novel_title = st.text_input("좋아하는 소설 제목을 입력하시오.")
st.write(f"검색 대상: {novel_title}")


random_data = [
    ["".join([random.choice(string.ascii_letters) for _ in range(random.randint(1, 16))]) for _ in range(16)],
    np.random.rand(16)
]
df = pd.DataFrame(zip(random_data[0], random_data[1]), columns=['name', 'acc'])
# df.columns = ['name', 'acc']
df.sort_values(by=['acc'], ascending=False)
print(df)


st.write("Here's our first attempt at using data to create a table:")
st.write(df)
