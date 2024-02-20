import os
import h5py
from torch.utils.data import Dataset
import mediapipe as mp
import numpy as np

class LandMarksDataset(Dataset):
    def __init__(self, dir_path, points_type, transform) -> None:
        super().__init__()

        data_path = os.path.join(dir_path, "train_data")
        with h5py.File(data_path, "r") as f:
            self.landmarks = f["landmarks"][:]
            self.labels = f["annotations"][:]
        
        if points_type == "contours":
            self.fct_type = extract_contours_landmarks
        else:
            raise ValueError("Invalid points type")

        self.transform = transform

    def __len__(self):
        return len(self.landmarks)

    def __getitem__(self, index):
        
        x = extract_contours_landmarks(self.landmarks[index][-1])
        y = self.labels[index]
        
        if self.transform:
            return self.transform(x), y
        else:
            return x, y


def extract_contours_landmarks(landmarks):
    """Returns the CONTOURS LANDMARKS"""
    ## get contours landmarks indices
    CONTOURS_INDICES = list(mp.solutions.face_mesh.FACEMESH_CONTOURS)
    CONTOURS_INDICES = np.unique(CONTOURS_INDICES)
    ## extract landmarks
    contours_landmarks = landmarks[CONTOURS_INDICES]
    return contours_landmarks
