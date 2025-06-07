import os
import numpy as np
import streamlit as st
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

import torch
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from PIL import Image

base_path = r"C:\Users\KML\Downloads\archive\The Welding Defect Dataset - v2\The Welding Defect Dataset - v2"
image_path = os.path.join(base_path, "train", "images")
label_path = os.path.join(base_path, "train", "labels")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
resnet.eval()

features, labels = [], []

for img_file in tqdm(os.listdir(image_path), desc="Processing Images"):
    if not img_file.endswith('.jpg'):
        continue
    img_path = os.path.join(image_path, img_file)
    label_file = os.path.join(label_path, img_file.replace('.jpg', '.txt'))
    if not os.path.exists(label_file):
        continue
    image = Image.open(img_path).convert('RGB')
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        feature = resnet(img_tensor).squeeze().numpy()
    with open(label_file, 'r') as f:
        line = f.readline()
        if not line.strip():
            continue
        class_id = int(line.strip().split()[0])
        features.append(feature)
        labels.append(class_id)

X = np.array(features)
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

st.title("Welding Defect Classifier")

uploaded_file = st.file_uploader("Upload a welding image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="Uploaded Image", use_column_width=True)
    img_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        test_feature = resnet(img_tensor).squeeze().numpy().reshape(1, -1)
    prediction = model.predict(test_feature)[0]
    if prediction == 0:
        st.success(" The welding process is CORRECT (No Defect)")
    else:
        st.error(f"The welding process is DEFECTIVE (Class ID: {prediction})")
import streamlit as st
import time

features, labels = [], []

image_files = [f for f in os.listdir(image_path) if f.endswith('.jpg')]

progress_bar = st.progress(0)
status_text = st.empty()

for i, img_file in enumerate(image_files):
    status_text.text(f"Processing image {i + 1} of {len(image_files)}: {img_file}")

    img_path = os.path.join(image_path, img_file)
    label_file = os.path.join(label_path, img_file.replace('.jpg', '.txt'))
    if not os.path.exists(label_file):
        continue
    image = Image.open(img_path).convert('RGB')
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        feature = resnet(img_tensor).squeeze().numpy()
    with open(label_file, 'r') as f:
        line = f.readline()
        if not line.strip():
            continue
        class_id = int(line.strip().split()[0])
        features.append(feature)
        labels.append(class_id)

    progress_bar.progress((i + 1) / len(image_files))

status_text.text("Feature extraction completed!")
progress_bar.empty()
features, labels = [], []

image_files = [f for f in os.listdir(image_path) if f.endswith('.jpg')]

progress_bar = st.progress(0)
status_text = st.empty()

for i, img_file in enumerate(image_files):
    status_text.text(f"Processing image {i + 1} of {len(image_files)}: {img_file}")

    img_path = os.path.join(image_path, img_file)
    label_file = os.path.join(label_path, img_file.replace('.jpg', '.txt'))
    if not os.path.exists(label_file):
        continue
    image = Image.open(img_path).convert('RGB')
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        feature = resnet(img_tensor).squeeze().numpy()
    with open(label_file, 'r') as f:
        line = f.readline()
        if not line.strip():
            continue
        class_id = int(line.strip().split()[0])
        features.append(feature)
        labels.append(class_id)

    progress_bar.progress((i + 1) / len(image_files))

status_text.text("Feature extraction completed!")
progress_bar.empty()
