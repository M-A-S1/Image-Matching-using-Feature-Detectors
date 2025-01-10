import os
import cv2
import numpy as np
from tkinter import Tk, filedialog, Label, Button, Entry, IntVar, Radiobutton, StringVar, messagebox
from matplotlib import pyplot as plt

data_folder_path = "Dataset" 

# Function to load images from a folder
def load_images_from_folder(folder_path):
    """Load all grayscale images from a folder."""
    images = []
    image_names = []
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
            image_names.append(filename)
    return images, image_names

# Function to detect keypoints based on the specified detector and compute SIFT descriptors
def detect_features(image, detector_type, sift):
    """
    Detect keypoints using the specified detector and compute descriptors using SIFT.
    Args:
        image: Input image
        detector_type: Feature detector type (SIFT, SURF, Harris, ORB)
        sift: SIFT object for descriptor computation
    Returns:
        keypoints: Detected keypoints
        descriptors: SIFT descriptors
    """
    if detector_type == "SIFT":
        keypoints = sift.detect(image, None)
    elif detector_type == "SURF":
        surf = cv2.xfeatures2d.SURF_create()
        keypoints = surf.detect(image, None)
    elif detector_type == "Harris":
        # Harris Corner Detection
        harris_corners = cv2.cornerHarris(np.float32(image), 2, 3, 0.04)
        keypoints = [cv2.KeyPoint(x, y, 1) for y, x in np.argwhere(harris_corners > 0.01 * harris_corners.max())]
    elif detector_type == "ORB":
        orb = cv2.ORB_create()
        keypoints = orb.detect(image, None)
    else:
        raise ValueError("Invalid detector type")

    # Compute descriptors using SIFT for all detectors
    keypoints, descriptors = sift.compute(image, keypoints)
    return keypoints, descriptors

# Function to match features using BFMatcher and Lowe's ratio test
def match_features(query_desc, train_desc):
    """
    Match features using BFMatcher and Lowe's ratio test.
    Args:
        query_desc: Descriptors of the query image
        train_desc: Descriptors of the database image
    Returns:
        good_matches: List of good matches after Lowe's ratio test
        total_matches: Total number of matches
    """
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)  # L2 norm for SIFT descriptors
    matches = bf.knnMatch(query_desc, train_desc, k=2)  # KNN matching with k=2
    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]  # Lowe's ratio test
    return good_matches, len(matches)

# Function to find the best match in the database folder
def find_best_match(query_image_path, folder_path, detector_type, threshold):
    """
    Find the best matching image in the database using feature matching and RANSAC.
    Args:
        query_image_path: Path to the query image
        folder_path: Path to the database folder
        detector_type: Feature detector type
        threshold: Minimum number of inliers to consider a match
    Returns:
        Best match details or 'No match found' if no sufficient match is found
    """
    sift = cv2.xfeatures2d.SIFT_create()  # Create SIFT object for descriptor computation
    query_image = cv2.imread(query_image_path, cv2.IMREAD_GRAYSCALE)
    query_kp, query_desc = detect_features(query_image, detector_type, sift)  # Detect query image features
    print ("The query descriptors are", query_desc.shape)
    database_images, image_names = load_images_from_folder(folder_path)  # Load all database images

    best_match = None
    best_image = None
    best_matches = None
    max_inliers = 0
    best_total_matches = 0
    best_homography = None

    # Iterate through all database images
    for img, name in zip(database_images, image_names):
        train_kp, train_desc = detect_features(img, detector_type, sift)  # Detect features in database image
        matches, total_matches = match_features(query_desc, train_desc)  # Match features

        # Apply RANSAC to compute inliers
        if len(matches) >= 4:  # Minimum 4 matches required for homography
            src_pts = np.float32([query_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([train_kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            if mask is not None:
                inliers = np.sum(mask)  # Count inliers
                if inliers > max_inliers:  # Update best match if current match has more inliers
                    max_inliers = inliers
                    best_total_matches = total_matches
                    best_match = name
                    best_image = img
                    best_matches = [m for m, valid in zip(matches, mask.ravel()) if valid]
                    best_homography = homography

    # Check if the best match meets the inliers threshold
    if max_inliers < threshold:
        return "No match found", None, None, None, None, None, None
    return best_match, max_inliers, query_image, best_image, best_matches, best_total_matches, best_homography, query_kp

# GUI Functions
def browse_query_image():
    """Browse and select the query image."""
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png *.jpeg")])
    query_image_path.set(file_path)

def match_image():
    """Perform image matching."""
    query_image = query_image_path.get()
    detector_type = detector_var.get()
    threshold = threshold_var.get()

    if not query_image:
        messagebox.showerror("Error", "Please provide the query image.")
        return

    result = find_best_match(query_image, data_folder_path, detector_type, threshold)

    if result[0] == "No match found":
        result_label.config(text="No match found!")

        # Display the query image with its features
        query_img = cv2.imread(query_image, cv2.IMREAD_GRAYSCALE)
        sift = cv2.xfeatures2d.SIFT_create()
        query_kp, _ = detect_features(query_img, detector_type, sift)

        query_img_with_features = cv2.drawKeypoints(
            query_img, query_kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )

        plt.figure(figsize=(10, 5))
        plt.imshow(query_img_with_features, cmap="gray")
        plt.title("Query Image with Detected Features")
        plt.axis("off")
        plt.show()

    else:
        best_match, inliers, query_img, best_img, best_matches, total_matches, homography, query_kp = result
        result_label.config(
            text=f"Best Match: {best_match}\nInliers: {inliers}\nTotal Matches: {total_matches}"
        )

        # Draw matches and display results
        sift = cv2.xfeatures2d.SIFT_create()
        best_kp, _ = detect_features(best_img, detector_var.get(), sift)

        if best_matches:
            matched_img = cv2.drawMatches(
                query_img, query_kp, best_img, best_kp, best_matches, None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )
            plt.figure(figsize=(10, 5))
            plt.imshow(matched_img, cmap="gray")
            plt.title("Matched Features")
            plt.axis("off")
            plt.show()
        else:
            messagebox.showinfo("No Valid Matches", "No valid matches could be drawn.")

# Tkinter GUI Setup
root = Tk()
root.title("Image Matching with Feature Detectors")

# GUI Variables
query_image_path = StringVar()
detector_var = StringVar(value="SIFT")
threshold_var = IntVar(value=10)

# GUI Layout
Label(root, text="Query Image:").grid(row=0, column=0, padx=10, pady=5, sticky="e")
Entry(root, textvariable=query_image_path, width=50).grid(row=0, column=1, padx=10, pady=5)
Button(root, text="Browse", command=browse_query_image).grid(row=0, column=2, padx=10, pady=5)

Label(root, text="Feature Detector:").grid(row=1, column=0, padx=10, pady=5, sticky="e")
Radiobutton(root, text="SIFT", variable=detector_var, value="SIFT").grid(row=1, column=1, sticky="w")
Radiobutton(root, text="SURF", variable=detector_var, value="SURF").grid(row=1, column=1)
Radiobutton(root, text="Harris Corner", variable=detector_var, value="Harris").grid(row=1, column=1, sticky="e")
Radiobutton(root, text="ORB", variable=detector_var, value="ORB").grid(row=1, column=2, sticky="w")

Label(root, text="Threshold:").grid(row=2, column=0, padx=10, pady=5, sticky="e")
Entry(root, textvariable=threshold_var, width=10).grid(row=2, column=1, padx=10, pady=5, sticky="w")

Button(root, text="Match", command=match_image).grid(row=3, column=0, columnspan=3, pady=20)

result_label = Label(root, text="", font=("Helvetica", 12), fg="blue")
result_label.grid(row=4, column=0, columnspan=3, pady=10)

# Start the Tkinter event loop
root.mainloop()
