"""
This module interprets the custom-defined raw-MNIST files that are expected to be in an adjacent directory
named "raw-dataset", and crucially defines a torch.Dataset subclass for accessing the data. Luckily, there's
a library ('mnist') someone else made that largely does the heavy lifting of interpreting the MNIST files' 
custom file layout which this module uses.
"""

from pathlib import Path

from mnist import MNIST as MNIST_DatasetParser
import numpy as np
import torch

class MNISTDataset(torch.utils.data.Dataset):
    
    raw_dataset_dir = Path(__file__).parent / "raw-dataset"
    # All of the images are 28x28 pixels.
    image_dimensions = (28, 28)

    def __init__(self, split: str):
        """
        Args:
            split: The relevant dataset-split, either "train" or "test".
        """
        
        assert split in ["train", "test"], f"split must be either 'train' or 'test', but received split: {split}."
        self.split = split 
        
        mnist_dataset_parser = MNIST_DatasetParser(path=self.raw_dataset_dir)
        
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
