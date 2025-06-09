import pandas as pd
import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel
import faiss
import requests
from PIL import Image
import io
import os
import logging
from tqdm import tqdm
from typing import List, Dict, Any
import json
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.image_dir = self.data_dir / "images"
        self.image_dir.mkdir(exist_ok=True)
        
        # Initialize CLIP model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Initialize FAISS index
        self.dimension = 512  # CLIP embedding dimension
        self.index = faiss.IndexFlatL2(self.dimension)
        
    def load_data(self) -> pd.DataFrame:
        """Load and combine all CSV files"""
        dfs = []
        for csv_file in self.data_dir.glob("*.csv"):
            logger.info(f"Loading {csv_file}")
            df = pd.read_csv(csv_file)
            dfs.append(df)
        
        combined_df = pd.concat(dfs, ignore_index=True)
        logger.info(f"Loaded {len(combined_df)} products")
        return combined_df
    
    def download_image(self, url: str, product_id: str) -> str:
        """Download image and save locally"""
        try:
            image_path = self.image_dir / f"{product_id}.jpg"
            if image_path.exists():
                return str(image_path)
            
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                image = Image.open(io.BytesIO(response.content))
                image.save(image_path)
                return str(image_path)
            return None
        except Exception as e:
            logger.error(f"Error downloading image {url}: {str(e)}")
            return None
    
    def get_image_embedding(self, image_path: str) -> np.ndarray:
        """Get CLIP embedding for an image"""
        try:
            image = Image.open(image_path)
            inputs = self.processor(
                images=image,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
                image_features = image_features / image_features.norm(dim=1, keepdim=True)
            
            return image_features.cpu().numpy().astype('float32')
        except Exception as e:
            logger.error(f"Error getting embedding for {image_path}: {str(e)}")
            return None
    
    def process_data(self):
        """Process all data and create FAISS index"""
        # Load data
        df = self.load_data()
        
        # Process each product
        embeddings = []
        metadata = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            # Download main image
            image_path = self.download_image(row['feature_image_s3'], row['product_id'])
            if not image_path:
                continue
            
            # Get image embedding
            embedding = self.get_image_embedding(image_path)
            if embedding is None:
                continue
            
            # Store embedding and metadata
            embeddings.append(embedding)
            metadata.append({
                'product_id': row['product_id'],
                'name': row['product_name'],
                'brand': row['brand'],
                'price': row['selling_price'],
                'category': row['category_id'],
                'image_path': image_path,
                'features': row['feature_list'],
                'style_attributes': row['style_attributes']
            })
        
        # Create FAISS index
        if embeddings:
            embeddings_array = np.vstack(embeddings)
            self.index.add(embeddings_array)
            
            # Save metadata
            with open(self.data_dir / "metadata.json", "w") as f:
                json.dump(metadata, f)
            
            # Save FAISS index
            faiss.write_index(self.index, str(self.data_dir / "faiss_index.bin"))
            
            logger.info(f"Processed {len(metadata)} products successfully")
        else:
            logger.error("No valid embeddings generated")

if __name__ == "__main__":
    processor = DataProcessor()
    processor.process_data() 