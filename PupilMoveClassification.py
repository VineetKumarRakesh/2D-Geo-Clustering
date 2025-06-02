import cv2
import dlib
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# -------------------------------
# Setup: load face detector and shape predictor.
# -------------------------------
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)


# -------------------------------
# Helper function: extract an eye region given landmark indices.
# -------------------------------
def get_eye_region(image, landmarks, eye_indices):
    """
    Given an image and facial landmarks, extract the eye region defined by eye_indices.

    Returns:
      eye_image: Cropped eye region.
      bbox: Tuple (x, y, w, h) for the bounding box of the eye region in the original image.
    """
    points = [(landmarks.part(i).x, landmarks.part(i).y) for i in eye_indices]
    points_array = np.array(points, dtype=np.int32)
    x, y, w, h = cv2.boundingRect(points_array)
    eye_image = image[y:y + h, x:x + w]
    return eye_image, (x, y, w, h)


# -------------------------------
# Helper function: detect the pupil within the eye region.
# -------------------------------
def detect_pupil(eye_image):
    """
    Detect the pupil in the eye image by thresholding for dark regions and using contour analysis.

    Returns:
      (cx, cy): The pupil center coordinates within the cropped eye_image,
      or None if the pupil is not detected.
    """
    gray_eye = cv2.cvtColor(eye_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray_eye, (7, 7), 0)
    # Threshold for dark regions (assuming the pupil is the darkest part)
    _, thresh = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Assume the largest contour is the pupil.
    pupil_contour = max(contours, key=cv2.contourArea)
    M = cv2.moments(pupil_contour)
    if M['m00'] == 0:
        return None
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    return (cx, cy)


# -------------------------------
# Helper function: classify the pupil position (heuristic method).
# -------------------------------
def classify_eye_position(pupil_coords, eye_bbox):
    """
    Classify the pupil position within the eye bounding box.

    Normalizes the pupil coordinates to [0, 1] and uses a dead zone around the center.

    Returns one of: "center", "left", "right", "top", "bottom".
    """
    (ex, ey, ew, eh) = eye_bbox
    cx, cy = pupil_coords

    # Normalize coordinates to [0, 1]
    norm_x = cx / ew
    norm_y = cy / eh

    dx = norm_x - 0.5
    dy = norm_y - 0.5

    dead_zone = 0.1
    if abs(dx) < dead_zone and abs(dy) < dead_zone:
        return "center"

    if abs(dx) >= abs(dy):
        return "left" if dx < 0 else "right"
    else:
        return "top" if dy < 0 else "bottom"


# -------------------------------
# Process a single image.
# -------------------------------
def process_image(image_path, annotated_output_folder):
    """
    Process an image to detect the face and right eye, locate the pupil, and classify its position.
    Annotates and saves the image with overlays.

    Returns:
      A dictionary with image details (paths, normalized coordinates, labels, etc.).
      Returns None if processing fails.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"[WARNING] Could not read image: {image_path}")
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if len(faces) == 0:
        print(f"[WARNING] No face detected in image: {image_path}")
        return None

    # Use the first detected face.
    face = faces[0]
    landmarks = predictor(gray, face)

    # Right eye indices for the 68-point model: 42-47.
    right_eye_indices = list(range(42, 48))
    eye_region, eye_bbox = get_eye_region(image, landmarks, right_eye_indices)

    pupil = detect_pupil(eye_region)
    if pupil is None:
        print(f"[WARNING] Pupil not detected in image: {image_path}")
        return None

    # Compute normalized pupil coordinates within the eye region.
    norm_x = pupil[0] / eye_bbox[2]
    norm_y = pupil[1] / eye_bbox[3]

    # Heuristic classification of pupil position.
    label = classify_eye_position(pupil, eye_bbox)

    # Global coordinates for annotation.
    global_pupil = (eye_bbox[0] + pupil[0], eye_bbox[1] + pupil[1])

    # Annotate the original image.
    annotated_img = image.copy()
    cv2.rectangle(annotated_img, (eye_bbox[0], eye_bbox[1]),
                  (eye_bbox[0] + eye_bbox[2], eye_bbox[1] + eye_bbox[3]), (0, 255, 0), 2)
    cv2.circle(annotated_img, global_pupil, 3, (0, 0, 255), -1)
    cv2.putText(annotated_img, label, (eye_bbox[0], eye_bbox[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Save the annotated image.
    os.makedirs(annotated_output_folder, exist_ok=True)
    filename = os.path.basename(image_path)
    annotated_path = os.path.join(annotated_output_folder, filename)
    cv2.imwrite(annotated_path, annotated_img)

    # Return collected details.
    return {
        "image_path": image_path,
        "annotated_path": annotated_path,
        "label": label,
        "norm_x": norm_x,
        "norm_y": norm_y,
        "eye_x": eye_bbox[0],
        "eye_y": eye_bbox[1],
        "eye_w": eye_bbox[2],
        "eye_h": eye_bbox[3],
        "pupil_x": pupil[0],
        "pupil_y": pupil[1]
    }


# -------------------------------
# Process all images in a folder (and sub-folders).
# -------------------------------
def cluster_images_by_eye_position(root_folder, annotated_output_folder):
    """
    Walk through root_folder recursively, process image files, and collect results.

    Returns:
      A list of dictionaries with per-image results.
    """
    results = []
    valid_exts = ('.png', '.jpg', '.jpeg')
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith(valid_exts):
                image_path = os.path.join(root, file)
                res = process_image(image_path, annotated_output_folder)
                if res:
                    results.append(res)
    return results


# -------------------------------
# Generate and save plots based on heuristic clustering.
# -------------------------------
def generate_plots(results_df, output_folder):
    """
    Generates and saves:
      - A bar plot of heuristic cluster counts.
      - A scatter plot of normalized pupil positions (color-coded by heuristic label).
    """
    os.makedirs(output_folder, exist_ok=True)

    # Bar Plot: Cluster Distribution.
    cluster_counts = results_df['label'].value_counts().sort_index()
    plt.figure(figsize=(8, 6))
    cluster_counts.plot(kind='bar', color='skyblue')
    plt.title("Heuristic Cluster Distribution")
    plt.xlabel("Pupil Position Cluster")
    plt.ylabel("Number of Images")
    plt.xticks(rotation=0)
    barplot_path = os.path.join(output_folder, "cluster_distribution.png")
    plt.tight_layout()
    plt.savefig(barplot_path)
    plt.close()
    print(f"Saved heuristic cluster bar plot to: {barplot_path}")

    # Scatter Plot: Normalized Pupil Coordinates.
    plt.figure(figsize=(8, 6))
    colors = {'center': 'blue', 'left': 'green', 'right': 'red', 'top': 'purple', 'bottom': 'orange'}
    for lbl in results_df['label'].unique():
        subset = results_df[results_df['label'] == lbl]
        plt.scatter(subset['norm_x'], subset['norm_y'],
                    label=lbl, color=colors.get(lbl, 'gray'),
                    alpha=0.6, edgecolors='w', s=80)
    plt.xlabel("Normalized Pupil X")
    plt.ylabel("Normalized Pupil Y")
    plt.title("Heuristic Clustering of Normalized Pupil Positions")
    plt.legend()
    plt.gca().invert_yaxis()  # Invert y-axis to match image coordinates.
    scatterplot_path = os.path.join(output_folder, "pupil_scatter.png")
    plt.tight_layout()
    plt.savefig(scatterplot_path)
    plt.close()
    print(f"Saved heuristic scatter plot to: {scatterplot_path}")


# -------------------------------
# Generate and save KMeans clustering plots.
# -------------------------------
def generate_kmeans_plots(results_df, output_folder, n_clusters=5):
    """
    Generates and saves a scatter plot of normalized pupil positions, color-coded by KMeans cluster.
    """
    os.makedirs(output_folder, exist_ok=True)

    plt.figure(figsize=(8, 6))
    cmap = plt.cm.get_cmap('viridis', n_clusters)
    for cluster in range(n_clusters):
        subset = results_df[results_df['kmeans_label'] == cluster]
        plt.scatter(subset['norm_x'], subset['norm_y'],
                    label=f'KMeans Cluster {cluster}',
                    color=cmap(cluster), alpha=0.6, edgecolors='w', s=80)
    plt.xlabel("Normalized Pupil X")
    plt.ylabel("Normalized Pupil Y")
    plt.title("KMeans Clustering of Normalized Pupil Positions")
    plt.legend()
    plt.gca().invert_yaxis()
    kmeans_scatter_path = os.path.join(output_folder, "kmeans_pupil_scatter.png")
    plt.tight_layout()
    plt.savefig(kmeans_scatter_path)
    plt.close()
    print(f"Saved KMeans scatter plot to: {kmeans_scatter_path}")


# -------------------------------
# Main function: process images, perform clustering (heuristic & KMeans), and save results.
# -------------------------------
def main():
    # Update the following paths as needed.
    input_folder = "colds_repositioned_images"  # Folder containing input images.
    annotated_folder = "07022024_Pupil_annotated_images"  # Folder to save annotated images.
    plots_output_folder = "07022024_plots"  # Folder to save plots.
    results_csv = os.path.join(plots_output_folder, "clustering_results.csv")   # CSV file for saving detailed results.

    print("Processing images...")
    results = cluster_images_by_eye_position(input_folder, annotated_folder)
    if not results:
        print("No images processed successfully.")
        return

    # Create a DataFrame and save the heuristic results to CSV.
    results_df = pd.DataFrame(results)
    results_df.to_csv(results_csv, index=False)
    print(f"Saved heuristic clustering results to CSV: {results_csv}")

    # Print heuristic cluster counts.
    counts = results_df['label'].value_counts().to_dict()
    print("Heuristic cluster counts:")
    for label, count in counts.items():
        print(f"  {label}: {count} images")

    # Generate and save heuristic plots.
    generate_plots(results_df, plots_output_folder)

    # --- Perform KMeans clustering on the normalized pupil positions ---
    n_clusters = 5  # You can adjust this number as needed.
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    # Fit KMeans on the (norm_x, norm_y) features.
    kmeans.fit(results_df[['norm_x', 'norm_y']])
    results_df['kmeans_label'] = kmeans.labels_
    inertia = kmeans.inertia_
    print(f"KMeans inertia: {inertia:.2f}")

    # Save the updated DataFrame (with KMeans labels) to CSV.
    results_df.to_csv(results_csv, index=False)
    print(f"Updated CSV saved with KMeans labels: {results_csv}")

    # Generate and save KMeans plots.
    generate_kmeans_plots(results_df, plots_output_folder, n_clusters=n_clusters)

    print("Clustering and result saving complete.")


if __name__ == '__main__':
    main()
