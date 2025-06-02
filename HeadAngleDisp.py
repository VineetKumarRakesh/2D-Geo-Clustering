import dlib
import cv2
import numpy as np
import os
import pandas as pd

# Load the dlib models
predictor_path = "shape_predictor_68_face_landmarks.dat"  # Update the path if needed
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# Define landmark indexes (Dlib index starts from 0)
point_37 = 36  # Left eye inner corner
point_46 = 45  # Right eye inner corner

# Paths
main_folder = "07022024_colds_repositioned_images"  # Update this path
output_folder = "07022024_colds_AngleDisp_Images"  # Folder where processed images will be stored
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Variables to compute the average eye positions
total_37 = np.array([0.0, 0.0])  # Sum of coordinates for point 37
total_46 = np.array([0.0, 0.0])  # Sum of coordinates for point 46
count = 0  # Counter for valid faces detected

# Traverse the folder and compute the average eye positions
for root, _, files in os.walk(main_folder):
    for file in files:
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(root, file)
            img = cv2.imread(img_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Detect faces
            faces = detector(gray)
            for face in faces:
                landmarks = predictor(gray, face)

                # Get coordinates of points 37 and 46
                p37 = np.array([landmarks.part(point_37).x, landmarks.part(point_37).y])
                p46 = np.array([landmarks.part(point_46).x, landmarks.part(point_46).y])

                # Accumulate sums
                total_37 += p37
                total_46 += p46
                count += 1

# Compute final average eye positions
if count > 0:
    avg_37 = total_37 / count
    avg_46 = total_46 / count
    print(f"Average Coordinate of Point 37 (x1, y1): {avg_37}")
    print(f"Average Coordinate of Point 46 (x2, y2): {avg_46}")
else:
    print("No faces detected in the images.")
    exit()

# Function to compute Euclidean distance between two points
def euclidean_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

# Function to compute angle between two lines given two pairs of points
def calculate_angle(p1, p2, ref1, ref2):
    """
    Computes the angle (in degrees) between:
    Line 1: p1 -> p2 (actual detected eye line)
    Line 2: ref1 -> ref2 (average eye line)
    """
    dY1, dX1 = p2[1] - p1[1], p2[0] - p1[0]
    dY2, dX2 = ref2[1] - ref1[1], ref2[0] - ref1[0]

    # Compute the angles of each line
    angle1 = np.arctan2(dY1, dX1)
    angle2 = np.arctan2(dY2, dX2)

    # Compute the difference in angles (convert to degrees)
    return np.degrees(angle1 - angle2)

# Step 2: Draw straight lines and compute displacement & angle
data = []  # Store displacement and angle for each image

for root, _, files in os.walk(main_folder):
    for file in files:
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(root, file)
            img = cv2.imread(img_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Detect faces
            faces = detector(gray)
            for face in faces:
                landmarks = predictor(gray, face)

                # Get actual detected eye coordinates
                p37 = np.array([landmarks.part(point_37).x, landmarks.part(point_37).y])
                p46 = np.array([landmarks.part(point_46).x, landmarks.part(point_46).y])

                # Compute midpoints of both lines
                avg_midpoint = ((avg_37[0] + avg_46[0]) / 2, (avg_37[1] + avg_46[1]) / 2)
                actual_midpoint = ((p37[0] + p46[0]) / 2, (p37[1] + p46[1]) / 2)

                # Draw reference (average) eye line - Green Line
                avg_37_int = tuple(avg_37.astype(int))
                avg_46_int = tuple(avg_46.astype(int))
                cv2.line(img, avg_37_int, avg_46_int, (0, 255, 0), 1)

                # Draw the actual detected eye line - Red Line
                p37_int = tuple(p37.astype(int))
                p46_int = tuple(p46.astype(int))
                cv2.line(img, p37_int, p46_int, (0, 0, 255), 1)

                # Compute displacement using midpoints
                avg_displacement = euclidean_distance(avg_midpoint, actual_midpoint)

                # Compute the actual angle difference between the lines
                angle = calculate_angle(p37, p46, avg_37, avg_46)

                # Save the results
                data.append([file, angle, avg_displacement])

                # Save the modified image
                output_path = os.path.join(output_folder, file)
                cv2.imwrite(output_path, img)

# Save displacement & angle data as CSV
df = pd.DataFrame(data, columns=["Filename", "Angle (degrees)", "Displacement (pixels)"])
csv_output_path = os.path.join(output_folder, "colds_AngleDisp_Data.csv")
df.to_csv(csv_output_path, index=False)

print(f"Processed images saved in '{output_folder}'")
print(f"Angle and displacement data saved in '{csv_output_path}'")
