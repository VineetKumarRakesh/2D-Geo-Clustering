import os
import cv2
import dlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# Initialize dlib's face detector and facial landmark predictor
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor(
    "shape_predictor_68_face_landmarks.dat")  # Replace with path to shape_predictor_68_face_landmarks.dat

# Input and output folders
input_folder = "colds"
output_folder = "Eye_Emo"
output_data_file = os.path.join(output_folder,"Eye_ExpFull_landmark_ratios.csv")
os.makedirs(output_folder, exist_ok=True)

# Open file to store results
with open(output_data_file, "w") as file:
    file.write("Image Name,Height,Width,Ratio\n")

# Process each image in the input folder and its subfolders
for root, _, files in os.walk(input_folder):
    for image_name in files:
        image_path = os.path.join(root, image_name)
        relative_path = os.path.relpath(image_path, input_folder)
        output_image_path = os.path.join(output_folder, relative_path)

        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_image_path), exist_ok=True)

        # Read the image
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            print(f"Unable to read {image_name}. Skipping.")
            continue

        # Convert to grayscale for detection
        if len(image.shape) == 3 and image.shape[2] == 3:  # Check if the image is colored
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif len(image.shape) == 3 and image.shape[2] == 4:  # Check for RGBA images
            gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        else:  # Image is already grayscale
            gray = image

        # Detect faces
        faces = face_detector(gray)
        if len(faces) == 0:
            print(f"No face detected in {image_name}. Skipping.")
            continue

        # Process the first detected face
        for face in faces:
            landmarks = landmark_predictor(gray, face)

            # Extract the required points for eyes
            point_37 = np.array([landmarks.part(36).x, landmarks.part(36).y])
            point_40 = np.array([landmarks.part(39).x, landmarks.part(39).y])

            point_41 = np.array([landmarks.part(40).x, landmarks.part(40).y])
            point_42 = np.array([landmarks.part(41).x, landmarks.part(41).y])

            point_38 = np.array([landmarks.part(37).x, landmarks.part(37).y])
            point_39 = np.array([landmarks.part(38).x, landmarks.part(38).y])

            # Calculate height as the distance between midpoints of 38,39 and 41,42
            midpoint_top = (point_38 + point_39) / 2
            midpoint_bottom = (point_41 + point_42) / 2
            height = np.linalg.norm(midpoint_top - midpoint_bottom)

            # Calculate width as the distance between points 36 and 39

            width = np.linalg.norm(point_37 - point_40)

            if height == 0:  # Avoid division by zero
                print(f"Invalid height calculation for {image_name}. Skipping.")
                continue
            ratio = width / height

            # Check for invalid ratio values
            if not np.isfinite(ratio):
                print(f"Invalid ratio for {image_name}. Skipping.")
                continue

            # Format values to 3 decimal points
            width = round(width, 3)
            height = round(height, 3)
            ratio = round(ratio, 3)
            print(f"Image={image_name}, Height={height}, Width={width}, Ratio={ratio}")
            # Write results to file
            with open(output_data_file, "a") as file:
                file.write(f"{relative_path},{height},{width},{ratio}\n")

            # Draw lines and annotate image
            cv2.line(image, tuple(point_37), tuple(point_40), (0, 255, 0), 2)  # Line for width
            cv2.line(image, tuple(midpoint_top.astype(int)), tuple(midpoint_bottom.astype(int)), (255, 0, 0),
                     2)  # Line for height
            cv2.putText(image, f"H: {height}, W: {width}, R: {ratio}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

            # Save the annotated image (handle grayscale and color images appropriately)
            if len(image.shape) == 2:  # Grayscale image
                cv2.imwrite(output_image_path, image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            else:  # Color or other formats
                cv2.imwrite(output_image_path, image)

            break  # Process only the first face per image

print("Processing complete. Results saved.")


# KMeans clustering function based on Ratio
def perform_kmeans_on_ratio(csv_file, output_dir, n_clusters=5):
    # Load the CSV data
    data = pd.read_csv(csv_file)

    # Check if data is empty
    if data.empty:
        print("No data available for clustering. Exiting.")
        return

    # Extract the Ratio feature
    ratios = data[["Ratio"]]

    # Check if ratios are valid
    if ratios.empty or ratios.isnull().all().all():
        print("No valid Ratio data for clustering. Exiting.")
        return

    # Scale the Ratio data using StandardScaler
    scaler = StandardScaler()
    scaled_ratios = scaler.fit_transform(ratios)

    # Check for infinite or NaN values
    if not np.all(np.isfinite(scaled_ratios)):
        print("Invalid scaled ratios detected. Exiting.")
        return

    # Elbow method to determine optimal number of clusters
    distortions = []
    K = range(1, 11)
    for k in K:
        kmeans = KMeans(
            n_clusters=k,
            init='k-means++',
            n_init=10,
            max_iter=300,
            tol=0.0001,
            random_state=42,
            algorithm='elkan'
        )
        kmeans.fit(scaled_ratios)
        distortions.append(kmeans.inertia_)

    # Plot the elbow graph
    plt.figure(figsize=(8, 6))
    plt.plot(K, distortions, marker='o')
    plt.title('Elbow Method for Optimal K')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Distortion')
    plt.grid(True)
    elbow_plot_path = os.path.join(output_dir, "elbow_method_plot.png")
    plt.savefig(elbow_plot_path)
    plt.close()

    print(f"Elbow plot saved at {elbow_plot_path}")

    # Perform KMeans clustering with specified hyperparameters
    # vineet: n_clusters=n_clusters or
    n_clusters=4
    kmeans = KMeans(
        n_clusters=n_clusters,
        init='k-means++',
        n_init=10,
        max_iter=300,
        tol=0.0001,
        random_state=42,
        algorithm='elkan'
    )
    data["Cluster"] = kmeans.fit_predict(scaled_ratios)

    # Calculate the mean ratio for each cluster
    cluster_means = data.groupby("Cluster")["Ratio"].mean()

    # Save clustered data
    clustered_csv = os.path.join(output_dir, "clustered_data_ratio.csv")
    data.to_csv(clustered_csv, index=False)

    # Calculate clustering evaluation metrics
    silhouette_avg = silhouette_score(scaled_ratios, kmeans.labels_)
    calinski_harabasz = calinski_harabasz_score(scaled_ratios, kmeans.labels_)
    davies_bouldin = davies_bouldin_score(scaled_ratios, kmeans.labels_)

    # Save metrics to a file
    metrics_file = os.path.join(output_dir, "clustering_metrics.txt")
    with open(metrics_file, "w") as file:
        file.write(f"Silhouette Score: {silhouette_avg:.4f}\n")
        file.write(f"Calinski-Harabasz Score: {calinski_harabasz:.4f}\n")
        file.write(f"Davies-Bouldin Score: {davies_bouldin:.4f}\n")

    # Create directories for each cluster
    for cluster in range(n_clusters):
        cluster_dir = os.path.join(output_dir, f"Cluster_{cluster}")
        os.makedirs(cluster_dir, exist_ok=True)

    # Move images to respective cluster folders
    for _, row in data.iterrows():
        image_name = row["Image Name"]
        cluster = row["Cluster"]
        source_path = os.path.join(output_folder, image_name)

        # Extract the relative path and preserve subfolder structure
        relative_path = os.path.relpath(source_path, output_folder)
        dest_folder = os.path.join(output_folder, f"Cluster_{cluster}", os.path.dirname(relative_path))
        dest_path = os.path.join(dest_folder, os.path.basename(relative_path))

        # Create the destination directory if it doesn't exist
        os.makedirs(dest_folder, exist_ok=True)

        if not os.path.exists(source_path):
            print(f"Warning: Source file {source_path} does not exist. Skipping.")
            continue

        print(f"Moving {source_path} to {dest_path}")
        os.rename(source_path, dest_path)

    # Annotate clustered images with landmarks
    for _, row in data.iterrows():
        image_name = row["Image Name"]
        cluster = row["Cluster"]
        image_path = os.path.join(output_dir, f"Cluster_{cluster}", image_name)
        if not os.path.exists(image_path):
            print(f"Warning: File {image_path} does not exist. Skipping.")
            continue
        image = cv2.imread(image_path)
        if image is None:
            print(f"Unable to read file {image_path}. Skipping.")
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_detector(gray)
        for face in faces:
            landmarks = landmark_predictor(gray, face)
            for i in range(68):
                x, y = landmarks.part(i).x, landmarks.part(i).y
                cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

            # Save the annotated image
            cv2.imwrite(image_path, image)
    # Plot clusters
    plt.figure(figsize=(8, 6))
    for cluster in range(n_clusters):
        cluster_data = data[data["Cluster"] == cluster]
        plt.scatter(cluster_data.index, cluster_data["Ratio"], label=f"Cluster {cluster}")

    plt.title("KMeans Clustering Based on Ratio")
    plt.xlabel("Image Index")
    plt.ylabel("Ratio")
    plt.legend()
    plt.grid(True)

    # Save the plot
    cluster_plot_path = os.path.join(output_dir, "clusters_plot_ratio.png")
    plt.savefig(cluster_plot_path)
    plt.close()

    print(f"Clustering based on Ratio complete. Results saved in {output_dir}")
    # Plot clusters with mean ratio
    plt.figure(figsize=(8, 6))
    for cluster in range(n_clusters):
        cluster_data = data[data["Cluster"] == cluster]
        cluster_mean = cluster_data["Ratio"].mean()
        plt.scatter([cluster] * len(cluster_data), cluster_data["Ratio"], alpha=0.5, label=f"Cluster {cluster}")
        plt.plot(cluster, cluster_mean, marker='o', markersize=10, label=f"Mean (Cluster {cluster})", color='red')

    plt.title("KMeans Clustering: Ratio vs. Mean")
    plt.xlabel("Cluster")
    plt.ylabel("Ratio")
    plt.xticks(range(n_clusters))
    plt.legend()
    plt.grid(True)

    # Save the plot
    cluster_plot_path = os.path.join(output_dir, "clusters_ratio_vs_mean_plot.png")
    plt.savefig(cluster_plot_path)
    plt.close()

    print(f"Clustering visualization saved at {cluster_plot_path}")


# Perform KMeans clustering based on Ratio
perform_kmeans_on_ratio(output_data_file, output_folder)
