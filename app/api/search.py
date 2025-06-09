from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
import faiss
import json
from pathlib import Path
from PIL import Image
import io
import torch
from transformers import CLIPProcessor, CLIPModel
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

# Initialize CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load FAISS index and metadata
data_dir = Path("data")
index = faiss.read_index(str(data_dir / "faiss_index.bin"))
with open(data_dir / "metadata.json", "r") as f:
    metadata = json.load(f)

@router.post("/visual")
async def visual_search(file: UploadFile = File(...), k: int = 10):
    """
    Perform visual search using an uploaded image
    """
    try:
        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Get image embedding
        inputs = processor(
            images=image,
            return_tensors="pt",
            padding=True
        ).to(device)
        
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(dim=1, keepdim=True)
        
        # Search in FAISS index
        query_vector = image_features.cpu().numpy().astype('float32')
        distances, indices = index.search(query_vector, k)
        
        # Get results
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(metadata):
                result = metadata[idx].copy()
                result['similarity_score'] = float(1 - distance)  # Convert distance to similarity
                results.append(result)
        
        return JSONResponse(content={
            "status": "success",
            "results": results
        })
        
    except Exception as e:
        logger.error(f"Error in visual search: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/similar/{product_id}")
async def similar_products(product_id: str, k: int = 10):
    """
    Find similar products to a given product ID
    """
    try:
        # Find product in metadata
        product_idx = None
        for idx, item in enumerate(metadata):
            if item['product_id'] == product_id:
                product_idx = idx
                break
        
        if product_idx is None:
            raise HTTPException(status_code=404, detail="Product not found")
        
        # Get product embedding from FAISS index
        query_vector = index.reconstruct(product_idx)
        query_vector = query_vector.reshape(1, -1).astype('float32')
        
        # Search in FAISS index
        distances, indices = index.search(query_vector, k + 1)  # +1 because the product itself will be in results
        
        # Get results (excluding the query product)
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx != product_idx and idx < len(metadata):
                result = metadata[idx].copy()
                result['similarity_score'] = float(1 - distance)
                results.append(result)
        
        return JSONResponse(content={
            "status": "success",
            "results": results
        })
        
    except Exception as e:
        logger.error(f"Error in similar products search: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 