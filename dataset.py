"""This module interprets the custom-defined raw-MNIST files that are expected to be in an adjacent directory
named "dataset", and crucially defines a torch.Dataset subclass for accessing the data."""

from pathlib import Path

from mnist import MNIST as MNIST_DatasetParser
import numpy as np


class MNISTDataset:
    
    raw_dataset_dir = Path(__file__) / "dataset"
    # All of the images are 28x28 pixels.
    image_dimensions = (28, 28)

    def __init__(self, split: str):
        """
        Args:
            split: The relevant dataset-split, either "train" or "test".
        """
        
        assert split in ["train", "test"], f"split must be either 'train' or 'test', but received split: {split}."
        self.split = split 
        
        mnist_dataset_parser = MNIST_DatasetParser(path="dataset")
        
        if self.split == 'train':
            images, labels = mnist_dataset_parser.load_training()
        else:
            images, labels = mnist_dataset_parser.load_testing()
        
        self.dataset_size = len(labels)
        
        self.images = np.array(images).reshape(self.dataset_size, *self.image_dimensions)
        self.labels = np.array(labels)

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        return (image, label)

mnist_dataset_train = MNISTDataset(split="train")
img, label = mnist_dataset_train[6]