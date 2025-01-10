# Image Matching using Feature Detectors

This Python project performs image matching using various feature detectors and descriptors such as SIFT, SURF, Harris Corner, and ORB. It utilizes OpenCV for computer vision tasks and Tkinter for a graphical user interface (GUI).

## Features
- **Feature Detectors:** SIFT, SURF, Harris Corner, ORB
- **Automatic Image Matching:** Uses feature detection and matching with Lowe's ratio test and RANSAC
- **Graphical User Interface (GUI):** Built with Tkinter for ease of use
- **Automatic Dataset Path:** The dataset folder path is hardcoded in the script for convenience.

## Requirements
Ensure you have the following packages installed:
- Python 3.6+
- OpenCV 3.4.2 (`opencv-python==3.4.2.16`, `opencv-contrib-python==3.4.2.16` for SURF support)
- NumPy
- Matplotlib
- Tkinter (built-in with Python)

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/M-A-S1/Image-Matching-using-Feature-Detectors.git
   ```
2. Navigate to the project folder:
   ```bash
   cd image-matching-feature-detectors
   ```
3. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: .\env\Scripts\activate
   ```
4. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Run the Python script:
   ```bash
   python app.py
   ```
2. Browse and select a query image.
3. The dataset folder path is hardcoded. Ensure the dataset is available at the specified location.
4. Select the desired feature detector and set the matching threshold.
5. Click the **Match** button to find the best match from the dataset.

## Dataset Setup
Ensure the dataset folder contains grayscale images. Update the dataset path in the code if necessary.

## Example
The GUI will display the matched features between the query image and the best-matched image from the dataset using feature detectors and descriptors.

## Contributing
Contributions are welcome! Feel free to submit pull requests or report issues.

## Acknowledgments
- OpenCV for the powerful computer vision tools.
- Python for providing a flexible programming language.

---
Feel free to modify the dataset path and adjust the threshold for better results!

