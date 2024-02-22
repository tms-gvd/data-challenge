import h5py
import os
import cv2
import json
import argparse
import numpy as np


def timesteps_to_classes(labels: np.array, max_length=37) -> np.array:
    new_labels = np.zeros(max_length)
    t1, t2, t3, t4 = labels
    new_labels[:t1] = 0
    new_labels[t1:t2] = 1
    new_labels[t2:t3] = 2
    new_labels[t3:t4] = 1
    new_labels[t4:] = 0
    return new_labels.astype(int)


class Preprocessor:
    def __init__(self, dir_path):

        self.dir_path = dir_path
        with h5py.File(os.path.join(dir_path, "train_data"), "r") as f:
            self.train_sequences = f["images"][:]
            self.train_labels = f["annotations"][:]
            self.landmarks = f["landmarks"][:]
        with h5py.File(os.path.join(dir_path, "test_data"), "r") as f:
            self.test_sequences = f["images"][:]

    def sequences_to_images(self):

        print("Preprocessing train data...")
        train_dir = os.path.join(self.dir_path, "images", "train")
        os.makedirs(train_dir, exist_ok=True)
        images_labels = []

        count = 0
        for i, sequence in enumerate(self.train_sequences):

            # save images and label the sequence i in dir_path / split_data / seq_i /
            folder_dir = os.path.join(train_dir, f"seq_{i}")
            os.makedirs(folder_dir, exist_ok=True)
            
            labels = self.train_labels[i]
            labels_image = timesteps_to_classes(labels)

            for j, image in enumerate(sequence):
                print(f"{i} / {j}", end="\r")

                image_path = os.path.join(folder_dir, f"image_{j}.png")
                image = cv2.resize(image, (224, 288))
                cv2.imwrite(
                    image_path,
                    cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                )

                images_labels.append(f"{i} ##$## {j} ##$## {image_path} ##$## {labels_image[j]}")
                count += 1

        train_path = os.path.join(train_dir, "images_labels.txt")
        with open(train_path, "w") as f:
            f.write("\n".join(images_labels))

        print("Preprocessing test data...")
        test_dir = os.path.join(self.dir_path, "images", "test")
        os.makedirs(test_dir, exist_ok=True)
        images_labels = []

        count = 0
        for i, sequence in enumerate(self.test_sequences):

            # save images and label the sequence i in dir_path / split_data / seq_i /
            folder_dir = os.path.join(test_dir, f"seq_{i}")
            os.makedirs(folder_dir, exist_ok=True)

            for j, image in enumerate(sequence):
                print(f"{i} / {j}", end="\r")

                image_path = os.path.join(folder_dir, f"image_{j}.png")
                image = cv2.resize(image, (224, 288))
                cv2.imwrite(
                    image_path,
                    cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                )

                images_labels.append(f"{i} ##$## {j} ##$## {image_path} ##$## ")
                count += 1

        test_path = os.path.join(test_dir, "images_labels.txt")
        with open(test_path, "w") as f:
            f.write("\n".join(images_labels))


if __name__ == "__main__":
    p = Preprocessor("data")
    p.sequences_to_images()