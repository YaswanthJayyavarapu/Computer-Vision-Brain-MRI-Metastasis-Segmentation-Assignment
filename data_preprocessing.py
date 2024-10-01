import os
import cv2
import numpy as np
import glob
from sklearn.model_selection import train_test_split
from albumentations import (
    Compose, HorizontalFlip, RandomRotate90, ElasticTransform, GridDistortion, RandomBrightnessContrast
)

# Path placeholders (update these according to your file structure)
IMAGE_PATH = 'path/to/your/images/folder'
MASK_PATH = 'path/to/your/masks/folder'
OUTPUT_PATH = 'path/to/your/output/folder'

# Apply CLAHE to enhance contrast in MRI images
def apply_clahe(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)

# Preprocess images and masks
def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read as grayscale
    image = apply_clahe(image)  # Apply CLAHE
    image = image / 255.0  # Normalize to [0,1]
    return image

def preprocess_mask(mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = mask / 255.0  # Normalize mask
    return mask

# Data Augmentation using Albumentations
def augment_image(image, mask):
    aug = Compose([
        HorizontalFlip(p=0.5),
        RandomRotate90(p=0.5),
        ElasticTransform(p=0.5),
        GridDistortion(p=0.5),
        RandomBrightnessContrast(p=0.5)
    ])
    augmented = aug(image=image, mask=mask)
    return augmented['image'], augmented['mask']

# Load images and masks
def load_data(image_path, mask_path):
    images = sorted(glob.glob(image_path + "/*.png"))  # Adjust extension if needed
    masks = sorted(glob.glob(mask_path + "/*.png"))  # Adjust extension if needed
    return images, masks

def split_dataset(images, masks):
    return train_test_split(images, masks, test_size=0.2, random_state=42)

if __name__ == "__main__":
    images, masks = load_data(IMAGE_PATH, MASK_PATH)

    # Split into train and test
    train_images, test_images, train_masks, test_masks = split_dataset(images, masks)

    # Apply preprocessing and save preprocessed images and masks
    for img_path, mask_path in zip(train_images, train_masks):
        img = preprocess_image(img_path)
        mask = preprocess_mask(mask_path)
        img_aug, mask_aug = augment_image(img, mask)

        # Save the augmented images and masks
        img_name = os.path.basename(img_path)
        mask_name = os.path.basename(mask_path)
        cv2.imwrite(f"{OUTPUT_PATH}/train_images/{img_name}", img_aug * 255)
        cv2.imwrite(f"{OUTPUT_PATH}/train_masks/{mask_name}", mask_aug * 255)

    print("Preprocessing complete.")
