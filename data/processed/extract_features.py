import pandas as pd
import numpy as np
import json
from PIL import Image
import requests
import io
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
import os

# Paths
CSV_PATH = 'data/processed/processed_metadata.csv'
IMG_COL = 'feature_image_s3'
OUT_PATH = 'data/processed/image_features.json'

# Load your CSV
print('Loading metadata...')
df = pd.read_csv(CSV_PATH)
# Limit to 1000 images for speed
image_urls = df[IMG_COL].dropna().tolist()[:1000]

# Load model
print('Loading ResNet50...')
model = resnet50(weights=ResNet50_Weights.DEFAULT)
model = torch.nn.Sequential(*(list(model.children())[:-1]))
model.eval()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

features = []
failed = 0
for idx, url in enumerate(image_urls):
    try:
        response = requests.get(url, timeout=10)
        img = Image.open(io.BytesIO(response.content)).convert('RGB')
        img_tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            feat = model(img_tensor).squeeze().numpy().tolist()
        if len(feat) != 2048:
            raise ValueError(f'Feature length {len(feat)} != 2048')
        features.append(feat)
    except Exception as e:
        print(f"[{idx+1}/{len(image_urls)}] Failed to process {url}: {e}")
        features.append([0.0]*2048)
        failed += 1
    if (idx+1) % 100 == 0:
        print(f"Processed {idx+1}/{len(image_urls)} images...")

print(f"Done. {failed} images failed out of {len(image_urls)}.")

# Save features
with open(OUT_PATH, 'w') as f:
    json.dump(features, f)
print(f"Saved features to {OUT_PATH}") 