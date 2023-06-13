import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

st.set_page_config(
    page_icon="🐶",
    page_title="패턴인식 현충일 보충",
    layout="wide",
)

#first title
col1, col2 = st.columns(2)
col1.title("6월 6일 현충일 보충강의")
col2.subheader("[202302801 김다운]")

#second title
st.title("Clustering - K-NN / K-Means")
st.title("")
st.header("**Clustering - 군집분석**")
st.markdown("")
st.markdown("'Clustering(군집분석)' :red[비지도학습](unsupervised learning)의 일종으로 기준이 없는 상태에서 주어진 데이터의 속성값들을 고려해 :blue[유사한 데이터끼리 그룹화를 시키는 학습 모델]을 말한다. 각 데이터의 유사성을 측정하여, 유사성이 높은 집단끼리  분류하고 군집간에 상이성을 규명하는 방법이다.")
st.markdown("")
img = Image.open("clustering.jpg")
st.image(
    img,
    caption="clustering 방법",
    width=800,
    channels="RGB"
)