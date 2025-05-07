import numpy as np

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