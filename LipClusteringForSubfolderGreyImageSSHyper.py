import os
import cv2
import dlib
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Define the paths
ROOT_FOLDER = "colds"
OUTPUT_FOLDER = "LipClustering1/output"
PLOTS_FOLDER = "LipClustering1/plots"
RESULTS_FILE = "LipClustering1/results.txt"



os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(PLOTS_FOLDER, exist_ok=True)

# Initialize Dlib's face detector and shape predictor
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Replace with the path to shape_predictor_68_face_landmarks.dat

def extract_ratio_and_draw(image_path, output_path, draw_lines=False):
    """Extract the ratio of Euclidean distances from the image and optionally draw lines."""
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_detector(gray)
    if len(faces) == 0:
        return None  # No face detected

    # Assuming the first detected face
    face = faces[0]
    landmarks = shape_predictor(gray, face)

    # Extract relevant landmark coordinates
    x63, y63 = landmarks.part(63).x, landmarks.part(63).y
    x67, y67 = landmarks.part(67).x, landmarks.part(67).y
    x61, y61 = landmarks.part(61).x, landmarks.part(61).y
    x65, y65 = landmarks.part(65).x, landmarks.part(65).y

    # Calculate Euclidean distances
    dist_63_67 = np.linalg.norm(np.array([x63, y63]) - np.array([x67, y67]))
    dist_61_65 = np.linalg.norm(np.array([x61, y61]) - np.array([x65, y65]))

    if dist_61_65 == 0:
        return None  # Avoid division by zero

    # Compute the ratio
    ratio = dist_63_67 / dist_61_65

    if draw_lines:
        # Draw lines
        cv2.line(image, (x63, y63), (x67, y67), (0, 255, 0), 2)
        cv2.line(image, (x61, y61), (x65, y65), (255, 0, 0), 2)

        # Write the lengths and ratio on the image
        cv2.putText(image, f"Length: {dist_63_67:.4f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(image, f"Height: {dist_61_65:.4f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(image, f"Ratio: {ratio:.4f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Save the image
        cv2.imwrite(output_path, image)

    return ratio

def process_images(root_folder, output_folder):
    """Process all images in the root folder and subfolders."""
    ratios = []
    image_paths = []
    processed_images = 0

    for subdir, _, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith(('jpg', 'jpeg', 'png')):
                image_path = os.path.join(subdir, file)
                output_path = os.path.join(output_folder, f"processed_{file}")
                draw_lines = processed_images < 50  # Draw lines on the first 50 images
                ratio = extract_ratio_and_draw(image_path, output_path, draw_lines=draw_lines)
                if ratio is not None:
                    ratios.append(ratio)
                    image_paths.append(image_path)
                    processed_images += 1

    return np.array(ratios).reshape(-1, 1), image_paths

# Process images
ratios, image_paths = process_images(ROOT_FOLDER, OUTPUT_FOLDER)

# Normalize the data
scaler = StandardScaler()
ratios_scaled = scaler.fit_transform(ratios)

# Apply KMeans clustering
n_clusters = 5  # Adjust the number of clusters as needed
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(ratios_scaled)

# Assign cluster labels to images
cluster_labels = kmeans.labels_

# Save the results
with open(RESULTS_FILE, "w") as f:
    for image_path, cluster in zip(image_paths, cluster_labels):
        f.write(f"Image: {image_path}, Cluster: {cluster}\n")

# Save plots
plt.figure()
plt.scatter(ratios_scaled, np.zeros_like(ratios_scaled), c=cluster_labels, cmap='viridis')
plt.title("Cluster Visualization")
plt.xlabel("Normalized Ratios")
plt.savefig(os.path.join(PLOTS_FOLDER, "cluster_visualization.png"))
plt.close()

plt.figure()
plt.hist(ratios, bins=20, color='blue', edgecolor='black')
plt.title("Histogram of Ratios")
plt.xlabel("Ratio")
plt.ylabel("Frequency")
plt.savefig(os.path.join(PLOTS_FOLDER, "ratios_histogram.png"))
plt.close()

print("Processing complete. Results, images, and plots have been saved.")
