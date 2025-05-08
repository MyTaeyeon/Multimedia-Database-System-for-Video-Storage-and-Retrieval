import numpy as np
import cv2

def calc_distance(vector1, vector2):
    """
    Compute the Euclidean distance between two vectors using NumPy.

    Args:
        vector1 (list, tuple, or np.ndarray): The first vector.
        vector2 (list, tuple, or np.ndarray): The second vector.

    Returns:
        float: The Euclidean distance between the two vectors.

    Raises:
        ValueError: If the vectors have different lengths.
    """
    vector1 = np.array(vector1)
    vector2 = np.array(vector2)
    
    if vector1.shape != vector2.shape:
        raise ValueError("Vectors must be of the same length.")
    
    distance = np.linalg.norm(vector1 - vector2)
    return distance

def calc_histogram(image):
    """
    Calculate the HSV color histogram of an image.

    Args:
        image (np.ndarray): The input image in BGR format.

    Returns:
        np.ndarray: The normalized histogram as a flattened array.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()