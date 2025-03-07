# -*- coding: utf-8 -*-
!kaggle datasets download -d ismailnasri20/driver-drowsiness-dataset-ddd

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models, datasets

import pandas as pd
import cv2
from PIL import Image
from tqdm import tqdm

from skimage.io import imread
from skimage.transform import rescale
from skimage.feature import hog
from skimage import exposure
from skimage import color

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

import os
from math import ceil
import seaborn as sns
import matplotlib.pyplot as plt
import warnings  # Import the warnings module

sns.set()
warnings.filterwarnings('ignore')  # This line now works correctly

# PyTorch-specific preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

!unzip /content/driver-drowsiness-dataset-ddd.zip

data_dir = "/content/Driver Drowsiness Dataset (DDD)"

drowsy_df = pd.DataFrame(columns=['filepaths', 'labels'])
non_drowsy_df = pd.DataFrame(columns=['filepaths', 'labels'])


drowsy_label = 'drowsy'
non_drowsy_label = 'non_drowsy'

drowsy_files = os.listdir(os.path.join(data_dir, "Drowsy"))
non_drowsy_files = os.listdir(os.path.join(data_dir, "Non Drowsy"))

drowsy_df['filepaths'] = [os.path.join(data_dir, "Drowsy", filename) for filename in drowsy_files]
drowsy_df['labels'] = drowsy_label

non_drowsy_df['filepaths'] = [os.path.join(data_dir, "Non Drowsy", filename) for filename in non_drowsy_files]
non_drowsy_df['labels'] = non_drowsy_label

combined_df = pd.concat([drowsy_df, non_drowsy_df], ignore_index=True)

combined_df = combined_df.sample(frac=1, random_state=42)
label_counts = combined_df['labels'].value_counts()

label_counts

combined_df.head(5)

plt.figure(figsize=(5, 5))
sns.barplot(
    x=combined_df['labels'].value_counts().index,
    y=combined_df['labels'].value_counts().values,
    palette="viridis"
)
plt.title('Data Distribution', fontsize=16)
plt.xlabel('Labels', fontsize=14)
plt.ylabel('Counts', fontsize=14)
plt.xticks(rotation=45)
plt.show()

def plot_sample_images(dataframe, num_rows, num_cols):

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(7, 7))
    fig.suptitle("Sample Images from Dataset", fontsize=16, y=0.92)

    for row in range(num_rows):
        for col in range(num_cols):
            img_index = row * num_cols + col
            if img_index < len(dataframe):

                img_path = dataframe.iloc[img_index]['filepaths']
                img_label = dataframe.iloc[img_index]['labels']


                img = Image.open(img_path)
                axes[row, col].imshow(img)
                axes[row, col].set_title(f'Label: {img_label}', fontsize=10)
                axes[row, col].axis('off')
            else:
                axes[row, col].axis('off')

    plt.tight_layout()
    plt.show()



num_rows = 3
num_cols = 3
plot_sample_images(combined_df, num_rows, num_cols)

from sklearn.model_selection import train_test_split

test_size = 0.2  # 20% for testing
val_size = 0.1   # 10% for validation

# Split the combined_df into train+val and test sets
train_val_df, test_df = train_test_split(combined_df, test_size=test_size, stratify=combined_df['labels'], random_state=42)

# Split the train+val into train and validation sets
train_df, val_df = train_test_split(train_val_df, test_size=val_size, stratify=train_val_df['labels'], random_state=42)

# Print the sizes of each set
print(f"Training set size: {len(train_df)}")
print(f"Validation set size: {len(val_df)}")
print(f"Testing set size: {len(test_df)}")

# Store the dataframes as CSV files
train_df.to_csv('train_data.csv', index=False)
val_df.to_csv('val_data.csv', index=False)
test_df.to_csv('test_data.csv', index=False)

# Optionally, if you want to save them as pickle files (if preferred over CSV):
# train_df.to_pickle('train_data.pkl')
# val_df.to_pickle('val_data.pkl')
# test_df.to_pickle('test_data.pkl')

from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image


class CustomDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):

        img_path = self.dataframe.iloc[idx]['filepaths']
        label = self.dataframe.iloc[idx]['labels']
        image = Image.open(img_path).convert("RGB")


        if self.transform:
            image = self.transform(image)

        return image, label


train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


train_dataset = CustomDataset(train_df, transform=train_transform)
val_dataset = CustomDataset(val_df, transform=test_transform)
test_dataset = CustomDataset(test_df, transform=test_transform)


train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)


print(f"Training dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")
print(f"Testing dataset size: {len(test_dataset)}")

import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


class CustomImageDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)

        self.transform = transform


        self.label_map = {'drowsy': 0, 'non_drowsy': 1}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 0]
        image = Image.open(img_name)

        label_str = self.data.iloc[idx, 1]
        label = self.label_map[label_str]

        if self.transform:
            image = self.transform(image)

        return image, label

base_model = models.densenet121(pretrained=True)


base_model.classifier = nn.Linear(in_features=1024, out_features=2)


model = base_model


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


train_dataset = CustomImageDataset(csv_file="/content/train_data.csv", transform=transform)
valid_dataset = CustomImageDataset(csv_file="/content/val_data.csv", transform=transform)
test_dataset = CustomImageDataset(csv_file="/content/test_data.csv", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


num_epochs = 6
best_val_loss = float('inf')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model.to(device)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0


    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} Training"):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()  # Zero the gradients
        outputs = model(inputs)  # Forward pass
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)  # Get the predicted class
        correct_preds += (predicted == labels).sum().item()
        total_preds += labels.size(0)

    # Calculate average training loss and accuracy
    avg_train_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct_preds / total_preds
    print(f"Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")

    # Validation Phase
    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    correct_preds = 0
    total_preds = 0

    with torch.no_grad():
        for inputs, labels in tqdm(valid_loader, desc=f"Epoch {epoch+1}/{num_epochs} Validation"):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_preds += (predicted == labels).sum().item()
            total_preds += labels.size(0)


    avg_val_loss = val_loss / len(valid_loader)
    val_accuracy = 100 * correct_preds / total_preds
    print(f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")


    if avg_val_loss < best_val_loss:
        print(f"Validation loss decreased ({best_val_loss:.4f} --> {avg_val_loss:.4f}). Saving model...")
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), "best_model.pth")

# Testing Phase
model.eval()  # Set the model to evaluation mode
test_loss = 0.0
correct_preds = 0
total_preds = 0

with torch.no_grad():  # No need to compute gradients during testing
    for inputs, labels in tqdm(test_loader, desc="Testing"):
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)  # Forward pass
        loss = criterion(outputs, labels)  # Compute loss

        test_loss += loss.item()
        _, predicted = torch.max(outputs, 1)  # Get the predicted class
        correct_preds += (predicted == labels).sum().item()
        total_preds += labels.size(0)

# Calculate average test loss and accuracy
avg_test_loss = test_loss / len(test_loader)
test_accuracy = 100 * correct_preds / total_preds
print(f"Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

"""Download"""

from google.colab import files
files.download("best_model.pth")

"""Reusing the model for later"""

from google.colab import files
uploaded = files.upload()

import torch
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn
import matplotlib.pyplot as plt


label_map = {0: 'drowsy', 1: 'non_drowsy'}

saved_model_path = "/content/best_model.pth"
test_image_path = "/content/drowsy9.jpg"
model = models.densenet121(pretrained=False)
model.classifier = nn.Linear(in_features=1024, out_features=2)
model.load_state_dict(torch.load(saved_model_path))
model.eval()


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

image = Image.open(test_image_path).convert("RGB")
image_for_display = image.copy()
image = transform(image).unsqueeze(0)


with torch.no_grad():
    outputs = model(image)
    _, predicted = torch.max(outputs, 1)
    predicted_label = label_map[predicted.item()]


plt.figure(figsize=(6, 6))
plt.imshow(image_for_display)
plt.title(f"Predicted Label: {predicted_label}", fontsize=16, color='blue')
plt.axis('off')
plt.show()
