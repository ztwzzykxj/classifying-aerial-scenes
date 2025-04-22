# imbalanced_dataset_tools.py
import os
import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from sklearn.utils.class_weight import compute_class_weight

# → 1. Calculate class weights for loss function
def get_class_weights(train_dataset):
    labels = [sample[1] for sample in train_dataset.samples]
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(labels),
        y=labels
    )
    return torch.tensor(class_weights, dtype=torch.float)

# → 2. Build WeightedRandomSampler for DataLoader
def get_weighted_sampler(train_dataset):
    labels = [sample[1] for sample in train_dataset.samples]
    class_counts = np.bincount(labels)
    class_weights = 1. / class_counts
    sample_weights = [class_weights[label] for label in labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    return sampler

# → 3. Class-aware transform (stronger augmentation for minority classes)
class CustomAugTransform:
    def __init__(self, class_name, minority_classes):
        if class_name in minority_classes:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(25),
                transforms.RandomPerspective(),
                transforms.ToTensor(),
                transforms.Normalize([0.5]*3, [0.5]*3)
            ])
        else:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5]*3, [0.5]*3)
            ])

    def __call__(self, img):
        return self.transform(img)

# → 4. Custom ImageFolder that applies per-class transforms
class ImbalancedImageFolder(ImageFolder):
    def __init__(self, root, minority_classes):
        super().__init__(root=root)
        self.minority_classes = minority_classes

    def __getitem__(self, index):
        path, target = self.samples[index]
        class_name = os.path.basename(os.path.dirname(path))
        sample = self.loader(path)
        transform = CustomAugTransform(class_name, self.minority_classes)
        sample = transform(sample)
        return sample, target

# → 5. Helper: get class distribution from folder
def get_class_distribution(folder):
    return {cls: len(os.listdir(os.path.join(folder, cls))) for cls in os.listdir(folder)}

def get_minority_classes(class_counts, threshold=300):
    return [cls for cls, count in class_counts.items() if count < threshold]