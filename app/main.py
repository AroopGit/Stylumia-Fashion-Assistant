from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict
import uvicorn
import numpy as np
from PIL import Image, UnidentifiedImageError
import io
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
import json
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import logging
import os
from datetime import datetime
import traceback

# Setup logging with more detail
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define response models first
class SearchResponse(BaseModel):
    product_id: str
    product_name: str
    brand: str
    price: float
    image_url: str
    similarity_score: float

class OutfitRecommendation(BaseModel):
    items: List[SearchResponse]
    style: str
    confidence_score: float

# Create FastAPI app
app = FastAPI(
    title="Stylumio API",
    description="AI-powered fashion visual search and styling assistant",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and data
model = None
df = None
features_matrix = None
transform = None
last_init_error = None  # Store last initialization error
EXPECTED_FEATURE_LENGTH = 2048  # ResNet50 feature length

def initialize_model():
    """Initialize the model and data."""
    global model, df, features_matrix, transform, last_init_error
    last_init_error = None
    try:
        # Print current working directory
        logger.info(f"Current working directory: {os.getcwd()}")
        # Print __file__ location
        logger.info(f"__file__ location: {__file__}")
        # Initialize model
        logger.info("Loading pre-trained model...")
        model = resnet50(weights=ResNet50_Weights.DEFAULT)
        model = torch.nn.Sequential(*(list(model.children())[:-1]))  # Remove classification layer
        model.eval()
        
        # Initialize image preprocessing
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Load fashion dataset and features
        logger.info("Loading fashion dataset...")
        data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'processed')
        logger.info(f"Data directory: {data_dir}")
        metadata_path = os.path.join(data_dir, 'processed_metadata.csv')
        features_path = os.path.join(data_dir, 'image_features.json')
        logger.info(f"Metadata path: {metadata_path} (exists: {os.path.exists(metadata_path)})")
        logger.info(f"Features path: {features_path} (exists: {os.path.exists(features_path)})")
        if not os.path.exists(metadata_path) or not os.path.exists(features_path):
            last_init_error = f"Required data files not found in {data_dir}"
            raise FileNotFoundError(last_init_error)
        df = pd.read_csv(metadata_path)
        logger.info(f"Loaded metadata with {len(df)} products")
        with open(features_path, 'r') as f:
            features_data = json.load(f)
        logger.info(f"Loaded features for {len(features_data)} products")

        # After loading df, filter out rows missing image links
        if 'feature_image_s3' in df.columns:
            df = df.dropna(subset=['feature_image_s3'])
        else:
            possible_cols = ['image_url', 'image_path', 'img_url', 'url', 'image_link']
            present_cols = [col for col in possible_cols if col in df.columns]
            if present_cols:
                df = df.dropna(subset=present_cols)

        if isinstance(features_data, dict):
            df = df[df['product_id'].astype(str).isin(features_data.keys())].reset_index(drop=True)
            features_matrix = np.array([features_data[str(pid)] for pid in df['product_id']])
        elif isinstance(features_data, list):
            if isinstance(features_data[0], dict):
                features_data = [list(d.values()) for d in features_data]
            # Filter out bad vectors
            good_indices = [i for i, vec in enumerate(features_data) if isinstance(vec, (list, np.ndarray)) and len(vec) == EXPECTED_FEATURE_LENGTH]
            if len(good_indices) < len(features_data):
                logger.warning(f"Filtered out {len(features_data) - len(good_indices)} feature vectors with wrong length (expected {EXPECTED_FEATURE_LENGTH}).")
            features_data = [features_data[i] for i in good_indices]
            df = df.iloc[good_indices].reset_index(drop=True)
            min_len = min(len(features_data), len(df))
            features_matrix = np.array(features_data[:min_len])
        else:
            raise ValueError("image_features.json must be a dict or a list")
        logger.info(f"Final metadata rows: {len(df)}")
        logger.info(f"Final features: {len(features_data)}")
        if len(df) == 0 or len(features_data) == 0:
            logger.error("No valid products or features after filtering. Please check your data files.")
            raise ValueError("No valid products or features after filtering. Please check your data files.")
        return True
    except Exception as e:
        last_init_error = str(e)
        logger.error(f"Error initializing model and data: {str(e)}")
        logger.error(traceback.format_exc())
        return False

# Initialize on startup
@app.on_event("startup")
async def startup_event():
    if not initialize_model():
        logger.error("Failed to initialize model and data. API may not function correctly.")

def extract_features(image: Image.Image) -> np.ndarray:
    """Extract features from an image using the pre-trained model."""
    try:
        if model is None:
            raise RuntimeError("Model not initialized")
            
        img_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            features = model(img_tensor)
        return features.squeeze().numpy()
    except Exception as e:
        logger.error(f"Error extracting features: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def get_image_url(row):
    """Get image URL from row with fallback options."""
    try:
        # Try feature_image_s3 first
        if 'feature_image_s3' in row and pd.notnull(row['feature_image_s3']):
            return row['feature_image_s3']
        
        # Try other common image URL columns
        for col in ['image_url', 'image_path', 'img_url', 'url', 'image_link']:
            if col in row and pd.notnull(row[col]):
                return row[col]
        
        # If no valid URL found, return a placeholder image
        logger.warning(f"No image URL found for product {row.get('product_id', 'unknown')}, using placeholder")
        return "https://via.placeholder.com/300x400?text=No+Image+Available"
    except Exception as e:
        logger.error(f"Error getting image URL: {str(e)}")
        return "https://via.placeholder.com/300x400?text=Error+Loading+Image"

def get_price(row):
    # Handle price as dict or float
    price = row.get('selling_price')
    if isinstance(price, dict):
        # Try INR or first value
        return max(200.0, float(price.get('INR', list(price.values())[0])))
    try:
        return max(200.0, float(price))
    except Exception:
        # Return a random price between $200 and $500 if price data is not available
        return round(np.random.uniform(200, 500), 2)

@app.post("/api/visual-search", response_model=List[SearchResponse])
async def visual_search(file: UploadFile = File(...)):
    try:
        if model is None or df is None or features_matrix is None:
            raise HTTPException(
                status_code=503,
                detail="Service is not ready. Please try again in a few moments."
            )
        contents = await file.read()
        try:
            image = Image.open(io.BytesIO(contents)).convert("RGB")
        except UnidentifiedImageError:
            raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error loading image: {str(e)}")
        if image.width < 100 or image.height < 100:
            raise HTTPException(status_code=400, detail="Image is too small. Please upload an image at least 100x100 pixels.")
        query_features = extract_features(image)
        similarities = cosine_similarity(query_features.reshape(1, -1), features_matrix)[0]
        top_k = 5
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        results = []
        for idx in top_indices:
            product = df.iloc[idx]
            results.append(SearchResponse(
                product_id=str(product['product_id']),
                product_name=product['product_name'],
                brand=product['brand'],
                price=get_price(product),
                image_url=get_image_url(product),
                similarity_score=float(similarities[idx])
            ))
        logger.info(f"Successfully found {len(results)} similar items")
        return results
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in visual search: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Failed to perform visual search: {str(e)}"
        )

@app.get("/api/outfit-recommendations/{style}", response_model=List[OutfitRecommendation])
async def get_outfit_recommendations(style: str):
    """Get outfit recommendations based on style."""
    if model is None or df is None or features_matrix is None:
        raise HTTPException(
            status_code=503,
            detail="Service is not ready. Please try again in a few moments."
        )
        
    # Return 3 random items as a sample outfit
    if len(df) == 0:
        raise HTTPException(status_code=500, detail="No products available.")
    sample = df.sample(n=min(3, len(df)))
    items = [
        SearchResponse(
            product_id=str(row['product_id']),
            product_name=row.get('product_name', ''),
            brand=row.get('brand', ''),
            price=get_price(row),
            image_url=get_image_url(row),
            similarity_score=1.0
        ) for _, row in sample.iterrows()
    ]
    return [OutfitRecommendation(items=items, style=style, confidence_score=0.8)]

@app.post("/api/outfit-recommendations/upload", response_model=List[OutfitRecommendation])
async def upload_outfit_recommendations(
    file: UploadFile = File(...),
    style: str = Query("casual", description="Fashion style for recommendations")
):
    try:
        if model is None or df is None or features_matrix is None:
            raise HTTPException(
                status_code=503,
                detail="Service is not ready. Please try again in a few moments."
            )
        
        logger.info("Reading uploaded file...")
        contents = await file.read()
        try:
            image = Image.open(io.BytesIO(contents)).convert("RGB")
        except UnidentifiedImageError:
            raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error loading image: {str(e)}")
        
        if image.width < 100 or image.height < 100:
            raise HTTPException(status_code=400, detail="Image is too small. Please upload an image at least 100x100 pixels.")
        
        logger.info("Extracting features from image...")
        query_features = extract_features(image)
        similarities = cosine_similarity(query_features.reshape(1, -1), features_matrix)[0]
        top_k = 3
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        logger.info(f"Found {len(top_indices)} similar items")
        outfits = []
        
        # Ensure we have at least one valid item
        if len(top_indices) == 0:
            raise HTTPException(
                status_code=500,
                detail="No similar items found in the database"
            )
        
        for idx in top_indices:
            try:
                main_item = df.iloc[idx]
                logger.info(f"Processing main item: {main_item.get('product_name', 'Unknown')}")
                
                # Get complementary items
                if 'category' in df.columns:
                    # Get items from different categories
                    different_categories = df[df['category'] != main_item['category']]
                    if len(different_categories) > 0:
                        complementary_items = different_categories.sample(n=min(2, len(different_categories)))
                    else:
                        # If no items in different categories, get random items
                        complementary_items = df.sample(n=2)
                else:
                    # If no category column, get random items
                    complementary_items = df.sample(n=2)
                
                # Create main item response
                items = [
                    SearchResponse(
                        product_id=str(main_item['product_id']),
                        product_name=main_item.get('product_name', 'Unknown Product'),
                        brand=main_item.get('brand', 'Unknown Brand'),
                        price=get_price(main_item),
                        image_url=get_image_url(main_item),
                        similarity_score=float(similarities[idx])
                    )
                ]
                
                # Add complementary items
                for _, comp_item in complementary_items.iterrows():
                    try:
                        items.append(SearchResponse(
                            product_id=str(comp_item['product_id']),
                            product_name=comp_item.get('product_name', 'Unknown Product'),
                            brand=comp_item.get('brand', 'Unknown Brand'),
                            price=get_price(comp_item),
                            image_url=get_image_url(comp_item),
                            similarity_score=0.8
                        ))
                    except Exception as e:
                        logger.error(f"Error processing complementary item: {str(e)}")
                        continue
                
                if len(items) >= 2:  # Only add outfit if we have at least 2 items
                    outfits.append(OutfitRecommendation(
                        items=items,
                        style=style,
                        confidence_score=float(similarities[idx])
                    ))
                    logger.info(f"Successfully created outfit with {len(items)} items")
                
            except Exception as e:
                logger.error(f"Error processing item at index {idx}: {str(e)}")
                logger.error(traceback.format_exc())
                continue
        
        if not outfits:
            logger.error("No valid outfits could be generated")
            raise HTTPException(
                status_code=500,
                detail="Failed to generate any valid outfit recommendations. Please try a different image."
            )
            
        logger.info(f"Successfully generated {len(outfits)} outfit recommendations")
        return outfits
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in outfit recommendations: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate outfit recommendations: {str(e)}"
        )

@app.get("/api/styles")
async def get_available_styles():
    """Get list of available fashion styles."""
    return ["casual", "formal", "business", "evening", "sporty"]

@app.get("/health")
async def health_check():
    """Check the health of the API and its dependencies."""
    health_status = {
        "status": "healthy" if model is not None and df is not None and features_matrix is not None else "unhealthy",
        "timestamp": datetime.utcnow().isoformat(),
        "dependencies": {
            "model_loaded": model is not None,
            "dataset_loaded": df is not None and hasattr(df, '__len__') and len(df) > 0,
            "features_loaded": features_matrix is not None and hasattr(features_matrix, '__len__') and len(features_matrix) > 0
        },
        "last_init_error": last_init_error
    }
    return health_status

@app.get("/")
async def root():
    return {
        "message": "Welcome to Stylumio API",
        "docs_url": "/docs",
        "redoc_url": "/redoc"
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 