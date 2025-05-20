import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt

@st.cache_data
def load_data():
    data = pd.read_csv("CC GENERAL.csv")
    return data

def preprocess(data):
    data = data.copy()
    data.drop(['CUST_ID'], axis=1, inplace=True)
    data.fillna(data.mean(numeric_only=True), inplace=True)
    return data

def scale_data(data):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    return data_scaled, scaler

def elbow_plot(data_scaled):
    wcss = []
    for k in range(2, 16):
        km = KMeans(n_clusters=k, random_state=0)
        km.fit(data_scaled)
        wcss.append(km.inertia_)
    fig, ax = plt.subplots()
    ax.plot(range(2, 16), wcss, marker='x')
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("WCSS (Inertia)")
    ax.set_title("Elbow Method")
    st.pyplot(fig)

def clustering_metrics(data_scaled):
    silhouette = []
    davies_bouldin = []
    for k in range(2, 16):
        km = KMeans(n_clusters=k, random_state=0)
        labels = km.fit_predict(data_scaled)
        silhouette.append(silhouette_score(data_scaled, labels))
        davies_bouldin.append(davies_bouldin_score(data_scaled, labels))
    fig, ax = plt.subplots()
    ax.plot(range(2, 16), silhouette, marker='o', label='Silhouette Score')
    ax.plot(range(2, 16), davies_bouldin, marker='*', label='Davies-Bouldin Index')
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Score")
    ax.set_title("Clustering Metrics")
    ax.legend()
    st.pyplot(fig)

def pca_plot(data_scaled, labels):
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(data_scaled)
    fig, ax = plt.subplots()
    scatter = ax.scatter(pcs[:,0], pcs[:,1], c=labels, cmap='tab10')
    legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
    ax.add_artist(legend1)
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.set_title("Cluster Visualization (PCA)")
    st.pyplot(fig)

def main():
    st.title("Credit Card Customer Clustering App")
    
    st.markdown("CC GENERAL.csv")
    data_load = st.button("Load Dataset")
    
    if data_load:
        data = load_data()
        st.write("Raw Data Sample:")
        st.dataframe(data.head())
        
        st.markdown("scaler")
        processed_data = preprocess(data)
        st.write("Data after preprocessing (no missing values, dropped ID):")
        st.dataframe(processed_data.head())
        
        st.markdown("data_scaled")
        data_scaled, scaler = scale_data(processed_data)
        st.write("Data has been scaled.")
        
        st.markdown("kmeans")
        st.write("Elbow Method Plot:")
        elbow_plot(data_scaled)
        st.write("Clustering Metrics:")
        clustering_metrics(data_scaled)
        
        st.markdown("### Step 5: Choose Number of Clusters")
        k = st.slider("Select number of clusters", min_value=2, max_value=15, value=4)
        
        st.markdown(f"### Step 6: Run KMeans with k = {k}")
        km = KMeans(n_clusters=k, random_state=0)
        labels = km.fit_predict(data_scaled)
        processed_data['Cluster'] = labels
        
        st.write("Cluster label distribution:")
        st.bar_chart(processed_data['Cluster'].value_counts())
        
        st.write("Cluster visualization (PCA):")
        pca_plot(data_scaled, labels)
        
        st.write("Cluster summary statistics:")
        st.dataframe(processed_data.groupby('Cluster').mean())
        
if __name__ == "__main__":
    main()