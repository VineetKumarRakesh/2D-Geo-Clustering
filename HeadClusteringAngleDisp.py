import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import os
import shutil

# Paths
csv_file = "07022024_colds_AngleDisp_Images/colds_AngleDisp_Data.csv"  # Update if needed
images_folder = "colds"  # Read images directly from colds_AngleDisp_Images
output_folder = "07022024_Head_Emo"
angle_folder = os.path.join(output_folder, "Angle")
displacement_folder = os.path.join(output_folder, "Displacement")
angle_displacement_folder = os.path.join(output_folder, "Angle_Displacement")

# Create folders for segregated results
os.makedirs(angle_folder, exist_ok=True)
os.makedirs(displacement_folder, exist_ok=True)
os.makedirs(angle_displacement_folder, exist_ok=True)

# Function to create subfolders for clusters
def create_cluster_folders(base_folder, n_clusters):
    for i in range(n_clusters):
        cluster_folder = os.path.join(base_folder, f"Cluster_{i}")
        os.makedirs(cluster_folder, exist_ok=True)

# Load the CSV file
df = pd.read_csv(csv_file)

# Extract relevant features
X_angle = df[['Angle (degrees)']].values
X_displacement = df[['Displacement (pixels)']].values
X_combined = df[['Angle (degrees)', 'Displacement (pixels)']].values

# Standardize the data
scaler = StandardScaler()
X_angle_scaled = scaler.fit_transform(X_angle)
X_displacement_scaled = scaler.fit_transform(X_displacement)
X_combined_scaled = scaler.fit_transform(X_combined)

# Function to copy images to respective cluster folders
def copy_images_to_clusters(cluster_data, cluster_id, folder_path):
    cluster_folder = os.path.join(folder_path, f"Cluster_{cluster_id}")
    os.makedirs(cluster_folder, exist_ok=True)

    for _, row in cluster_data.iterrows():
        image_name = row['Filename']
        source_path = os.path.join(images_folder, image_name)
        destination_path = os.path.join(cluster_folder, image_name)
        if os.path.exists(source_path):
            shutil.copy(source_path, destination_path)

# Function to perform KMeans clustering, save segregated data, and compute evaluation metrics
def perform_kmeans_clustering(X, feature_name, x_label, y_label=None, n_clusters=3, folder_path=None):
    create_cluster_folders(folder_path, n_clusters)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=500)
    clusters = kmeans.fit_predict(X)

    # Add cluster labels to dataframe
    df[f'{feature_name} Cluster'] = clusters

    # Save segregated data and copy images
    for cluster_id in range(n_clusters):
        cluster_data = df[df[f'{feature_name} Cluster'] == cluster_id]
        cluster_data.to_csv(os.path.join(folder_path, f"{feature_name}_Cluster_{cluster_id}.csv"), index=False)
        copy_images_to_clusters(cluster_data, cluster_id, folder_path)

    # Compute evaluation metrics
    silhouette = silhouette_score(X, clusters)
    davies_bouldin = davies_bouldin_score(X, clusters)
    calinski_harabasz = calinski_harabasz_score(X, clusters)

    # Plot clustering results
    plt.figure(figsize=(8, 6))
    if y_label is None:  # Single feature clustering (1D)
        plt.scatter(
            range(len(X)), X.flatten(), c=clusters, cmap="viridis"
        )
        plt.xlabel("Data Index")
        plt.ylabel(x_label)
    else:  # Two-feature clustering (2D)
        plt.scatter(
            X[:, 0], X[:, 1], c=clusters, cmap="viridis"
        )
        plt.xlabel(x_label)
        plt.ylabel(y_label)

    plt.title(f"KMeans Clustering Based on {feature_name}")
    plt.colorbar(label="Cluster")

    # Save the plot
    plot_path = os.path.join(folder_path, f"KMeans_Clustering_{feature_name}.png")
    plt.savefig(plot_path)
    plt.close()

    # Evaluation metrics
    inertia = kmeans.inertia_
    print(f"Inertia for {feature_name}: {inertia}")
    print(f"Silhouette Score for {feature_name}: {silhouette}")
    print(f"Davies-Bouldin Index for {feature_name}: {davies_bouldin}")
    print(f"Calinski-Harabasz Score for {feature_name}: {calinski_harabasz}")

    return inertia, silhouette, davies_bouldin, calinski_harabasz

# Perform KMeans clustering on Angle only
angle_inertia, angle_silhouette, angle_db, angle_ch = perform_kmeans_clustering(
    X_angle_scaled, "Angle", "Angle (degrees)", n_clusters=3, folder_path=angle_folder
)

# Perform KMeans clustering on Displacement only
displacement_inertia, displacement_silhouette, displacement_db, displacement_ch = perform_kmeans_clustering(
    X_displacement_scaled, "Displacement", "Displacement (pixels)", n_clusters=3, folder_path=displacement_folder
)

# Perform KMeans clustering on both Angle & Displacement
angle_displacement_inertia, angle_displacement_silhouette, angle_displacement_db, angle_displacement_ch = perform_kmeans_clustering(
    X_combined_scaled, "Angle_Displacement", "Angle (degrees)", "Displacement (pixels)", n_clusters=5, folder_path=angle_displacement_folder
)

# Save updated dataframe with cluster labels
csv_output_path = os.path.join(output_folder, "KMeans_Clustering_Results.csv")
df.to_csv(csv_output_path, index=False)

# Save evaluation metrics
metrics = {
    "Feature": ["Angle", "Displacement", "Angle_Displacement"],
    "Inertia": [angle_inertia, displacement_inertia, angle_displacement_inertia],
    "Silhouette Score": [angle_silhouette, displacement_silhouette, angle_displacement_silhouette],
    "Davies-Bouldin Index": [angle_db, displacement_db, angle_displacement_db],
    "Calinski-Harabasz Score": [angle_ch, displacement_ch, angle_displacement_ch]
}
metrics_df = pd.DataFrame(metrics)
metrics_csv_path = os.path.join(output_folder, "Clustering_Evaluation_Metrics.csv")
metrics_df.to_csv(metrics_csv_path, index=False)

# Display results
print(f"KMeans clustering results saved in '{csv_output_path}'")
print(f"Evaluation metrics saved in '{metrics_csv_path}'")
