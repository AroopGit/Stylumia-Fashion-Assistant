from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
import json
from pathlib import Path
import logging
from typing import List, Dict, Any
from collections import Counter

router = APIRouter()
logger = logging.getLogger(__name__)

# Load metadata
data_dir = Path("data")
with open(data_dir / "metadata.json", "r") as f:
    metadata = json.load(f)

def extract_style_attributes(products: List[Dict[str, Any]]) -> Dict[str, int]:
    """Extract and count style attributes from products"""
    style_attrs = []
    for product in products:
        if isinstance(product.get('style_attributes'), dict):
            style_attrs.extend(product['style_attributes'].keys())
    return dict(Counter(style_attrs))

def extract_features(products: List[Dict[str, Any]]) -> Dict[str, int]:
    """Extract and count features from products"""
    features = []
    for product in products:
        if isinstance(product.get('features'), list):
            features.extend(product['features'])
    return dict(Counter(features))

@router.get("/categories")
async def category_trends():
    """
    Get trending categories and their attributes
    """
    try:
        # Group products by category
        categories = {}
        for product in metadata:
            category = product['category']
            if category not in categories:
                categories[category] = []
            categories[category].append(product)
        
        # Analyze trends for each category
        trends = {}
        for category, products in categories.items():
            trends[category] = {
                "count": len(products),
                "style_attributes": extract_style_attributes(products),
                "features": extract_features(products),
                "avg_price": np.mean([float(p['price']) for p in products if p.get('price')])
            }
        
        return JSONResponse(content={
            "status": "success",
            "trends": trends
        })
        
    except Exception as e:
        logger.error(f"Error in category trends: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/attributes")
async def attribute_trends():
    """
    Get trending style attributes across all categories
    """
    try:
        # Extract all style attributes
        all_attributes = extract_style_attributes(metadata)
        
        # Get top attributes
        top_attributes = dict(sorted(
            all_attributes.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10])
        
        return JSONResponse(content={
            "status": "success",
            "trending_attributes": top_attributes
        })
        
    except Exception as e:
        logger.error(f"Error in attribute trends: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/features")
async def feature_trends():
    """
    Get trending features across all categories
    """
    try:
        # Extract all features
        all_features = extract_features(metadata)
        
        # Get top features
        top_features = dict(sorted(
            all_features.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10])
        
        return JSONResponse(content={
            "status": "success",
            "trending_features": top_features
        })
        
    except Exception as e:
        logger.error(f"Error in feature trends: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 