import os
import h5py
from torch.utils.data import Dataset
from torchvision import transforms
import mediapipe as mp
import numpy as np
import json
import cv2
import torch


class ImagesFromJSON(Dataset):
    def __init__(self, path_to_config, transform_x=None) -> None:
        super().__init__()

        with open(path_to_config, "r") as f:
            self.paths = json.load(f)

        # set up basic transform
        if transform_x is None:
            self.transform_x = transforms.Compose([transforms.ToTensor()])
        else:
            self.transform_x = transform_x

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image_path = self.paths[str(index)]["image_path"]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR) / 255.0

        label = self.paths[str(index)]["labels"]

        return self.transform_x(image), torch.Tensor(label).long()


def extract_contours_landmarks(landmarks):
    """Returns the CONTOURS LANDMARKS"""
    ## get contours landmarks indices
    CONTOURS_INDICES = list(mp.solutions.face_mesh.FACEMESH_CONTOURS)
    CONTOURS_INDICES = np.unique(CONTOURS_INDICES)
    ## extract landmarks
    contours_landmarks = landmarks[CONTOURS_INDICES]
    return contours_landmarks


def timesteps_to_classes(labels: np.array, max_length=37) -> np.array:
    new_labels = np.zeros(max_length)
    t1, t2, t3, t4 = labels
    new_labels[:t1] = 0
    new_labels[t1:t2] = 1
    new_labels[t2:t3] = 2
    new_labels[t3:t4] = 1
    new_labels[t4:] = 0
    return new_labels.astype(int)


def timesteps_to_one_hot(labels, max_length=37):
    new_labels = np.zeros(max_length)
    new_labels[labels - 1] = 1
    # assert np.all(new_labels.sum(axis=1) == 4.)
    return new_labels.astype(int)
