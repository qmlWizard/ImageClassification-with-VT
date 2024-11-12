import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

class DatasetGenerator:
    def __init__(self, dataset_name, root='./data', batch_size=32, subset_size=None, image_size=(224, 224)):
        self.dataset_name = dataset_name.lower()
        self.root = root
        self.batch_size = batch_size
        self.subset_size = subset_size
        self.image_size = image_size
        self.transform = self._get_transform()
        self.train_loader, self.test_loader = self._prepare_data_loaders()

    def _get_transform(self):
        """Define transformation for the input based on the dataset and image size."""
        return transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) if self.dataset_name in ['cifar10', 'imagenet'] else transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def _get_dataset(self, train=True):
        """Load the specified dataset."""
        if self.dataset_name == 'cifar10':
            return datasets.CIFAR10(root=self.root, train=train, download=True, transform=self.transform)
        elif self.dataset_name == 'imagenet':
            return datasets.ImageNet(root=self.root, split='train' if train else 'val', download=True, transform=self.transform)
        elif self.dataset_name == 'mnist':
            return datasets.MNIST(root=self.root, train=train, download=True, transform=self.transform)
        else:
            raise ValueError("Dataset not supported. Choose from 'cifar10', 'imagenet', or 'mnist'.")

    def _prepare_data_loaders(self):
        """Prepare the data loaders with an optional subset."""
        # Load datasets
        train_dataset = self._get_dataset(train=True)
        test_dataset = self._get_dataset(train=False)
        
        # Apply subset if specified
        if self.subset_size:
            subset_indices = list(range(self.subset_size))
            train_dataset = Subset(train_dataset, subset_indices)
            test_dataset = Subset(test_dataset, subset_indices)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        return train_loader, test_loader

    def datasummary(self):
        """Print dataset summary including preprocessing, features, and feature vector."""
        print(f"Dataset: {self.dataset_name.capitalize()}")
        
        # Show preprocessing transformations
        print("Preprocessing:")
        for t in self.transform.transforms:
            print(f"  - {t}")
        
        # Display feature information
        sample_data, _ = next(iter(self.train_loader))
        print(f"Features Shape: {sample_data.shape}")
        print(f"Feature Vector (Flattened) Size: {sample_data.view(sample_data.size(0), -1).shape[-1]}")

    def get_loaders(self):
        """Return the train and test data loaders."""
        return self.train_loader, self.test_loader
