import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(
    page_icon="🐶",
    page_title="패턴인식 현충일 보충"
)

st.header("K-NN(K-Nearest Neighbor)")
st.markdown("")
st.markdown("새로운 데이터를 특정 데이터를 기준으로 주변에서 가장 가까운 K개의 데이터를 보고, 유사 속성으로 분류하여 사전에 지정한 K개 만큼 묶는 기법")

#데이터셋 로드
iris=load_iris()
X=iris.data
y=iris.target

#데이터셋 분리
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

#KNN 알고리즘 적용
knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)

#예측결과
y_pred=knn.predict(X_test)
print("Predictions : ",y_pred)

#정확도 계산
accuracy=np.mean(y_pred==y_test)
print("Accuracy : ",accuracy)

#분류데이터 시각화
plt.scatter(X[:,0],X[:,1],c=y,cmap='viridis')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.title('Iris Classification')
#plt.show()
st.pyplot(plt)