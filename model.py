# Authors: 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from preprocessing import Preprocessor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

annotations_path = r'annotations.csv'
images_dir = r'./Images'
annotations = pd.read_csv(annotations_path)

data = []
for index, row in annotations.iterrows():
    for fundus in ['Left_Fundus', 'Right_Fundus']:
        img_path = f"{images_dir}/{row[fundus]}"  
        processed_image = Preprocessor(img_path)
        if processed_image is not None:
            data.append((processed_image, row['Age']))

X = np.array([i[0] for i in data]).reshape(-1, 256, 256, 1)  
y = np.array([i[1] for i in data])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.12, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=(10/12), random_state=42)

X_train_tensor = torch.tensor(X_train).float().to(device)
y_train_tensor = torch.tensor(y_train).float().to(device)
X_val_tensor = torch.tensor(X_val).float().to(device)
y_val_tensor = torch.tensor(y_val).float().to(device)
X_test_tensor = torch.tensor(X_test).float().to(device)
y_test_tensor = torch.tensor(y_test).float().to(device)

X_train_tensor = torch.tensor(X_train).float().permute(0, 3, 1, 2)
y_train_tensor = torch.tensor(y_train).float()
X_val_tensor = torch.tensor(X_val).float().permute(0, 3, 1, 2)
y_val_tensor = torch.tensor(y_val).float()
X_test_tensor = torch.tensor(X_test).float().permute(0, 3, 1, 2)
y_test_tensor = torch.tensor(y_test).float()

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)
print("Preparing to train")

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 128 * 128, 128)  
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = CNNModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

num_epochs = 10  #
for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0
    for images, ages in train_loader:
        images, ages = images.to(device), ages.to(device)

        
        outputs = model(images)
        loss = criterion(outputs, ages.view(-1, 1))

        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()
        
model.eval()
total_val_loss = 0
with torch.no_grad():
    for images, ages in val_loader:
        images, ages = images.to(device), ages.to(device)
        outputs = model(images)
        loss = criterion(outputs, ages.view(-1, 1))
        total_val_loss += loss.item()

print(f"Epoch [{epoch+1}/{num_epochs}], "
        f"Train Loss: {total_train_loss / len(train_loader):.4f}, "
        f"Validation Loss: {total_val_loss / len(val_loader):.4f}")

model.eval()
total_test_loss = 0
with torch.no_grad():
    for images, ages in test_loader:
        images, ages = images.to(device), ages.to(device)
        outputs = model(images)
        loss = criterion(outputs, ages.view(-1, 1))
        total_test_loss += loss.item()

print(f"Test Loss: {total_test_loss / len(test_loader):.4f}")
