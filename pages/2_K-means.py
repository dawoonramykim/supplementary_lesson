import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(
    page_icon="🐶",
    page_title="패턴인식 현충일 보충"
)

st.header("K-means")
st.markdown("")
st.markdown("각 군집에 할당된 포인트들의 :red[평균 좌표를 이용]해서 중심점을 반복적으로 업데이트하며 군집을 형성하는 알고리즘이다.")

st.sidebar.markdown("K-means의 :red[변수] ")
k = st.sidebar.slider('How many K do you want?',1,6,3)
st.write("The current k is ", k)

data=np.random.randn(100,2)

def initialize_centroids(data,k):
    centroids=data[np.random.choice(data.shape[0],k,replace=False)]
    return centroids

def assign_clsuters(data,centroids):
    distance=np.sqrt(((data-centroids[:,np.newaxis])**2).sum(axis=2))
    clusters=np.argmin(distance,axis=0)
    return clusters

def update_centroids(data,clusters,k):
    new_centroids=np.array([data[clusters==i].mean(axis=0) for i in range(k)])
    return new_centroids

def k_means(data,k,max_iterations=100):
    centroids=initialize_centroids(data,k)
    prev_centroids=centroids.copy()

    for _ in range(max_iterations):
        clusters=assign_clsuters(data,centroids)
        centroids=update_centroids(data,clusters,k)

        if np.all(prev_centroids==centroids):
            break
        prev_centroids=centroids.copy()

    return centroids,clusters

centroids,clusters=k_means(data,k)

for i in range(k):
    cluster_data=data[clusters==i]
    plt.scatter(cluster_data[:,0],cluster_data[:,1],label=f"cluster{i+1}")


plt.scatter(centroids[:,0],centroids[:,1],marker="x",color="k",s=100,label="centroids")
plt.legend()
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("K-means Clustering")
#plt.show()
st.pyplot(plt)
