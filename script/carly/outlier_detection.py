import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

def identify_outliers(df: pd.DataFrame):
    """
    Identify outliers with DBSCAN after scaling and decomposition. 
    Drop outliers.

    Args:
        df (pd.Dataframe): Dataframe with only numerical columns
    """
    df_numerical = df.select_dtypes(include=['number'])

    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_numerical)


    pca = PCA()
    _ = pca.fit(df_scaled)

    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)

    n_components_90 = np.argmax(cumulative_variance >= 0.9) + 1  # +1 because index is 0-based
    print(f"Number of components to retain 90% variance: {n_components_90}")

    pca = PCA(n_components=n_components_90)  # Use the number of components determined earlier
    pca_data = pca.fit_transform(df_scaled)

        # Use NearestNeighbors to find the k-nearest distances
    k = 5  # Common choice for DBSCAN; can adjust based on data characteristics
    neighbors = NearestNeighbors(n_neighbors=k)
    neighbors.fit(pca_data)
    distances, indices = neighbors.kneighbors(pca_data)
    # Get the distances to the k-th nearest neighbor
    k_distances = np.sort(distances[:, k-1])

    # Plot k-distance graph
    plt.figure(figsize=(10, 6))
    plt.plot(k_distances)
    plt.title('K-Distance Graph')
    plt.xlabel('Data Points sorted by Distance to their {}-th Nearest Neighbor'.format(k))
    plt.ylabel('Distance')
    plt.grid()
    plt.show()

    # Initialize DBSCAN to detect anomalies
    db = DBSCAN(eps=1, min_samples=5) # 5/9
    db.fit(df_scaled)

    labels = db.labels_
    df['cluster'] = labels

    # Identify anomalies (labeled as -1 by DBSCAN)
    anomalies = df[df['cluster'] == -1]
    print(f"Anomalies found: {len(anomalies)}")

    return df

