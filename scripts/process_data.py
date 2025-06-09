import pandas as pd
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import requests
from io import BytesIO
import os
from tqdm import tqdm
import json
from pathlib import Path
import logging
from sklearn.preprocessing import LabelEncoder
import re
from concurrent.futures import ThreadPoolExecutor
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FashionDataProcessor:
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.processed_dir = self.data_dir / "processed"
        self.processed_dir.mkdir(exist_ok=True)
        
        # Initialize model and transforms
        logger.info("Initializing ResNet50 model...")
        self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.model = torch.nn.Sequential(*(list(self.model.children())[:-1]))
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Initialize encoders
        self.brand_encoder = LabelEncoder()
        self.category_encoder = LabelEncoder()
        
    def download_image(self, url, max_retries=3):
        """Download image from URL with retries."""
        for attempt in range(max_retries):
            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                return Image.open(BytesIO(response.content))
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.warning(f"Failed to download image from {url} after {max_retries} attempts: {str(e)}")
                    return None
                time.sleep(1)  # Wait before retrying
    
    def extract_features(self, image):
        """Extract features from an image using the pre-trained model."""
        try:
            img_tensor = self.transform(image).unsqueeze(0)
            with torch.no_grad():
                features = self.model(img_tensor)
            return features.squeeze().numpy()
        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            return None
    
    def clean_text(self, text):
        """Clean and normalize text data."""
        if not isinstance(text, str):
            return ""
        # Remove special characters and extra whitespace
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def process_metadata(self, df):
        """Process and clean metadata fields."""
        logger.info("Processing metadata...")
        
        # Clean text fields
        text_columns = ['meta_info', 'product_name', 'description', 'feature_list']
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].apply(self.clean_text)
        
        # Process price data
        if 'selling_price' in df.columns:
            def extract_price(price_str):
                try:
                    if isinstance(price_str, dict):
                        return float(price_str['INR'])
                    elif isinstance(price_str, str):
                        # Extract number from string like "{'INR': 474848.9539}"
                        import re
                        match = re.search(r"'INR':\s*([\d.]+)", price_str)
                        if match:
                            return float(match.group(1))
                    return float(price_str)
                except (ValueError, TypeError, KeyError):
                    return None
            
            df['price'] = df['selling_price'].apply(extract_price)
        
        # Process style attributes
        if 'style_attributes' in df.columns:
            df['style_attributes'] = df['style_attributes'].apply(
                lambda x: x if isinstance(x, dict) else {}
            )
        
        # Encode categorical variables
        if 'brand' in df.columns:
            df['brand_encoded'] = self.brand_encoder.fit_transform(df['brand'].fillna('unknown'))
        if 'category_id' in df.columns:
            df['category_encoded'] = self.category_encoder.fit_transform(df['category_id'].astype(str))
        
        return df
    
    def process_images(self, df, max_workers=4):
        """Process images in parallel and extract features."""
        logger.info("Processing images...")
        
        def process_single_image(row):
            try:
                # Download main image
                image = self.download_image(row['feature_image_s3'])
                if image is None:
                    return None
                
                # Extract features
                features = self.extract_features(image)
                if features is None:
                    return None
                
                return {
                    'product_id': row['product_id'],
                    'features': features.tolist(),
                    'image_url': row['feature_image_s3']
                }
            except Exception as e:
                logger.error(f"Error processing image for product {row['product_id']}: {str(e)}")
                return None
        
        # Process images in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(tqdm(
                executor.map(process_single_image, [row for _, row in df.iterrows()]),
                total=len(df)
            ))
        
        # Filter out None results and create features DataFrame
        image_features = pd.DataFrame([r for r in results if r is not None])
        return image_features
    
    def save_processed_data(self, df, image_features):
        """Save processed data to files."""
        logger.info("Saving processed data...")
        
        # Save processed metadata
        df.to_csv(self.processed_dir / "processed_metadata.csv", index=False)
        
        # Save image features
        with open(self.processed_dir / "image_features.json", 'w') as f:
            json.dump(image_features.to_dict('records'), f)
        
        # Save encoders
        np.save(self.processed_dir / "brand_encoder.npy", self.brand_encoder.classes_)
        np.save(self.processed_dir / "category_encoder.npy", self.category_encoder.classes_)
        
        logger.info("Data processing completed successfully!")
    
    def process_dataset(self):
        """Main method to process the entire dataset."""
        try:
            # Load both datasets
            logger.info("Loading datasets...")
            dresses_df = pd.read_csv(self.data_dir / "dresses_bd_processed_data.csv")
            jeans_df = pd.read_csv(self.data_dir / "jeans_bd_processed_data.csv")
            
            # Add category information
            dresses_df['category'] = 'dress'
            jeans_df['category'] = 'jeans'
            
            # Combine datasets
            df = pd.concat([dresses_df, jeans_df], ignore_index=True)
            logger.info(f"Combined dataset size: {len(df)} rows")
            
            # Process metadata
            df = self.process_metadata(df)
            
            # Process images and extract features
            image_features = self.process_images(df)
            
            # Save processed data
            self.save_processed_data(df, image_features)
            
        except Exception as e:
            logger.error(f"Error processing dataset: {str(e)}")
            raise

if __name__ == "__main__":
    processor = FashionDataProcessor()
    processor.process_dataset() 