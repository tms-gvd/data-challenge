import h5py
import os
import cv2
import numpy as np
import json
from PIL import Image


class Preprocessor:
    def __init__(self, dir_path):

        self.dir_path = dir_path
        with h5py.File(os.path.join(dir_path, "train_data"), "r") as f:
            self.train_sequences = f["images"][:]
            self.train_labels = f["annotations"][:]
        with h5py.File(os.path.join(dir_path, "test_data"), "r") as f:
            self.test_sequences = f["images"][:]

    def sequences_to_images(self, transform_x=None, transform_y=None):

        print("Preprocessing train data...")
        train_dir = os.path.join(self.dir_path, "split_data", "train")
        os.makedirs(train_dir, exist_ok=True)
        images_labels = {}
        
        count = 0
        for i, sequence in enumerate(self.train_sequences):

            # save images and label the sequence i in dir_path / split_data / seq_i /
            folder_dir = os.path.join(train_dir, f"seq_{i}")
            os.makedirs(folder_dir, exist_ok=True)

            for j, image in enumerate(sequence):
                print(f"{i} / {j}", end="\r")

                image_path = os.path.join(folder_dir, f"image_{j}.png")

                if transform_x:
                    image = transform_x(image)
                cv2.imwrite(image_path, image)

                labels = self.train_labels[i]
                if transform_y:
                    labels = transform_y(labels)
                
                images_labels[count] = {
                    "image_path": image_path,
                    "labels": labels.tolist(),
                }
                count += 1

        images_labels_path = os.path.join(train_dir, "images_labels.json")
        with open(images_labels_path, "w") as f:
            json.dump(images_labels, f, indent=4)

        print("Preprocessing test data...")
        test_dir = os.path.join(self.dir_path, "split_data", "test")
        os.makedirs(test_dir, exist_ok=True)
        images_labels = {}
        count = 0

        for i, sequence in enumerate(self.test_sequences):

            folder_dir = os.path.join(test_dir, f"seq_{i}")
            os.makedirs(folder_dir, exist_ok=True)

            for j, image in enumerate(sequence):
                print(f"{i} / {j}", end="\r")

                image_path = os.path.join(folder_dir, f"image_{j}.png")

                if transform_x:
                    image = transform_x(image)
                cv2.imwrite(image_path, image)

                images_labels[count] = {
                    "image_path": image_path
                }
                count += 1

        images_labels_path = os.path.join(test_dir, "images_labels.json")
        with open(images_labels_path, "w") as f:
            json.dump(images_labels, f, indent=4)

        return images_labels_path


### TRANSFORMS ON IMAGES ###


def resize_image(image: np.array) -> Image:
    img = Image.fromarray(image)
    return img.resize((224, 288))


def gray_to_rgb(image: Image) -> np.array:
    return np.array(image.convert("RGB"))


### TRANSFORMS ON IMAGES ###


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
