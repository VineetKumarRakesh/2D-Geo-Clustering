import dlib
import cv2
import numpy as np
import os

# Load the dlib models
predictor_path = "shape_predictor_68_face_landmarks.dat"  # Update if needed
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# Define landmark index for the nose tip
point_34 = 33  # Nose tip

# Paths
input_folder = "colds"  # Update this path
output_folder = "07022024_colds_repositioned_images"  # Folder where repositioned images will be stored

# Create output directory if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Step 1: Compute the Average Nose Tip Coordinate
total_nose_tip = np.array([0.0, 0.0])
count = 0

# Traverse the folder and compute the average nose tip position
for root, _, files in os.walk(input_folder):
    for file in files:
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(root, file)
            img = cv2.imread(img_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Detect faces
            faces = detector(gray)
            for face in faces:
                landmarks = predictor(gray, face)
                p34 = np.array([landmarks.part(point_34).x, landmarks.part(point_34).y])  # Nose tip
                total_nose_tip += p34
                count += 1

# Compute final average nose tip coordinate
if count > 0:
    avg_nose_tip = total_nose_tip / count
    print(f"Average Nose Tip Coordinate (x, y): {avg_nose_tip}")
else:
    print("No faces detected in the images.")
    exit()

# Step 2: Reposition Images Based on the Average Nose Tip Coordinate
for root, _, files in os.walk(input_folder):
    for file in files:
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(root, file)
            img = cv2.imread(img_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Detect faces
            faces = detector(gray)
            for face in faces:
                landmarks = predictor(gray, face)

                # Get current nose tip position
                p34 = np.array([landmarks.part(point_34).x, landmarks.part(point_34).y])  # Nose tip

                # Compute translation (shift)
                dx = avg_nose_tip[0] - p34[0]  # Shift in X
                dy = avg_nose_tip[1] - p34[1]  # Shift in Y

                # Apply translation to reposition nose tip
                M = np.float32([[1, 0, dx], [0, 1, dy]])
                repositioned_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

                # Save the repositioned image
                output_path = os.path.join(output_folder, file)
                cv2.imwrite(output_path, repositioned_img)

print(f"Repositioned images saved in '{output_folder}'.")
