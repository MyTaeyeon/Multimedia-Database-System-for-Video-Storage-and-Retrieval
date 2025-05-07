import numpy as np

def compute_distance(vector1, vector2):
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