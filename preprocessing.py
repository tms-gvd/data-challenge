import h5py
import os
import cv2
import json
import argparse


class Preprocessor:
    def __init__(self, dir_path):

        self.dir_path = dir_path
        with h5py.File(os.path.join(dir_path, "train_data"), "r") as f:
            self.train_sequences = f["images"][:]
            self.train_labels = f["annotations"][:]
            self.landmarks = f["landmarks"][:]
        with h5py.File(os.path.join(dir_path, "test_data"), "r") as f:
            self.test_sequences = f["images"][:]

    def sequences_to_images(self, transform_y=None):

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
                image = cv2.resize(image, (224, 288))
                cv2.imwrite(image_path, cv2.cvtColor(image, cv2.COLOR_GRAY2RGB))

                labels = self.train_labels[i]
                if transform_y is not None:
                    labels = transform_y(labels)

                images_labels[count] = {
                    "image_path": image_path,
                    "labels": labels.tolist(),
                }
                count += 1

        train_path = os.path.join(train_dir, "images_labels.json")
        with open(train_path, "w") as f:
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

                cv2.imwrite(image_path, image)

                images_labels[count] = {"image_path": image_path}
                count += 1

        test_path = os.path.join(test_dir, "images_labels.json")
        with open(test_path, "w") as f:
            json.dump(images_labels, f, indent=4)

        return train_path, test_path


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--dir_path", type=str, default="data")
    args = argparser.parse_args()
    
    p = Preprocessor(args.dir_path)
    train_path, test_path = p.sequences_to_images()
    print(f"Train data saved at {train_path}")
    print(f"Test data saved at {test_path}")