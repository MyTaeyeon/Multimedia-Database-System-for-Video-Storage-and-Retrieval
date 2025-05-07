import numpy as np

def compute_distance(vector1, vector2, metric='euclidean'):
    """
    Compute the distance between two vectors.

    Parameters:
        vector1 (array-like): The first vector.
        vector2 (array-like): The second vector.
        metric (str): The distance metric to use. Options are 'euclidean', 'manhattan', or 'cosine'.

    Returns:
        float: The computed distance.
    """
    vector1 = np.array(vector1)
    vector2 = np.array(vector2)

    if metric == 'euclidean':
        return np.linalg.norm(vector1 - vector2)
    elif metric == 'manhattan':
        return np.sum(np.abs(vector1 - vector2))
    elif metric == 'cosine':
        dot_product = np.dot(vector1, vector2)
        norm1 = np.linalg.norm(vector1)
        norm2 = np.linalg.norm(vector2)
        return 1 - (dot_product / (norm1 * norm2))
    else:
        raise ValueError(f"Unsupported metric '{metric}'. Use 'euclidean', 'manhattan', or 'cosine'.")
    
def load_features(file_path):
    """
    Load features from a file.

    Parameters:
        file_path (str): Path to the file containing features.

    Returns:
        np.ndarray: A numpy array of features.
    """
    try:
        return np.load(file_path)
    except Exception as e:
        raise IOError(f"Error loading features from {file_path}: {e}")