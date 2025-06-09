from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
import faiss
import json
from pathlib import Path
import logging
from typing import List, Dict, Any

router = APIRouter()
logger = logging.getLogger(__name__)

# Load FAISS index and metadata
data_dir = Path("data")
index = faiss.read_index(str(data_dir / "faiss_index.bin"))
with open(data_dir / "metadata.json", "r") as f:
    metadata = json.load(f)

def get_category_products(category: str, k: int = 5) -> List[Dict[str, Any]]:
    """Get random products from a specific category"""
    category_products = [item for item in metadata if item['category'] == category]
    if not category_products:
        return []
    indices = np.random.choice(len(category_products), min(k, len(category_products)), replace=False)
    return [category_products[i] for i in indices]

@router.post("/recommend")
async def recommend_outfit(product_id: str):
    """
    Generate outfit recommendations based on a product
    """
    try:
        # Find the product
        product = None
        for item in metadata:
            if item['product_id'] == product_id:
                product = item
                break
        
        if not product:
            raise HTTPException(status_code=404, detail="Product not found")
        
        # Get product category
        category = product['category']
        
        # Define complementary categories based on the product category
        complementary_categories = {
            "dresses": ["shoes", "accessories"],
            "tops": ["bottoms", "shoes", "accessories"],
            "bottoms": ["tops", "shoes", "accessories"],
            "shoes": ["dresses", "tops", "bottoms"],
            "accessories": ["dresses", "tops", "bottoms"]
        }
        
        # Generate outfit recommendations
        outfit = {
            "main_item": product,
            "complementary_items": []
        }
        
        # Add complementary items
        for comp_category in complementary_categories.get(category, []):
            items = get_category_products(comp_category, k=2)
            if items:
                outfit["complementary_items"].extend(items)
        
        return JSONResponse(content={
            "status": "success",
            "outfit": outfit
        })
        
    except Exception as e:
        logger.error(f"Error in outfit recommendation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/trending")
async def trending_outfits():
    """
    Get trending outfit combinations
    """
    try:
        # Get random products from different categories
        categories = ["dresses", "tops", "bottoms", "shoes", "accessories"]
        trending_outfits = []
        
        for _ in range(5):  # Generate 5 trending outfits
            outfit = {
                "main_item": get_category_products(np.random.choice(categories), k=1)[0],
                "complementary_items": []
            }
            
            # Add complementary items
            for category in categories:
                if category != outfit["main_item"]["category"]:
                    items = get_category_products(category, k=1)
                    if items:
                        outfit["complementary_items"].extend(items)
            
            trending_outfits.append(outfit)
        
        return JSONResponse(content={
            "status": "success",
            "trending_outfits": trending_outfits
        })
        
    except Exception as e:
        logger.error(f"Error in trending outfits: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 