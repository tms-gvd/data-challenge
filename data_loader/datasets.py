import os
import h5py
from torch.utils.data import Dataset
from torchvision import transforms
import mediapipe as mp
import numpy as np
import json
import cv2
import torch


class ImagesFromTXT(Dataset):
    def __init__(self, path_to_config, transform_x=None) -> None:
        super().__init__()
        
        with open(path_to_config, "r") as f:
            lines = f.readlines()
        seq_ids, img_ids, paths, labels = [], [], [], []
        
        for line in lines:
            seq_id, img_id, path, label = line.split(" ##$## ")
            seq_ids.append(int(seq_id))
            img_ids.append(int(img_id))
            paths.append(path)
            labels.append(int(label))
        
        self.seq_ids = seq_ids
        self.img_ids = img_ids
        self.paths = paths
        self.labels = torch.Tensor(labels).long()

        # set up basic transform
        if transform_x is None:
            self.transform_x = transforms.Compose([transforms.ToTensor()])
        else:
            self.transform_x = transform_x

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image = cv2.imread(self.paths[index], cv2.IMREAD_COLOR) / 255.0

        return (self.seq_ids[index], self.img_ids[index]), self.transform_x(image), self.labels[index]


class ImagesClassif(Dataset):
    def __init__(self, dir_path) -> None:
        super().__init__()

        self.dir_path = dir_path
        with h5py.File(os.path.join(dir_path, "train_data"), "r") as f:
            train_sequences = f["images"][:]
            _, _, h, w = train_sequences.shape
            self.train_sequences = train_sequences.reshape(-1, h, w)
            train_labels = f["annotations"][:]
        
        self.train_labels = torch.from_numpy(np.array([timesteps_to_classes(label) for label in train_labels]).flatten())
        
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((288, 224)),
                transforms.Grayscale(num_output_channels=3),  # Convert grayscale to RGB
                transforms.ToTensor(),
            ]
        )
    
    def __len__(self):
        return len(self.train_sequences)
    
    def __getitem__(self, index):
        return self.transform(self.train_sequences[index]), self.train_labels[index]


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
