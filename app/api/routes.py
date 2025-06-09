from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse
from typing import List, Optional
import logging

from app.services.visual_search import VisualSearchService
from app.services.outfit_recommender import OutfitRecommenderService
from app.core.config import settings
from app.schemas.search import SearchResponse, OutfitRecommendation

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/search", response_model=SearchResponse)
async def visual_search(
    image: UploadFile = File(...),
    limit: int = 10,
    min_confidence: float = 0.7
):
    """
    Perform visual search on uploaded fashion image
    """
    try:
        # Validate file type
        if not image.filename.lower().endswith(tuple(settings.ALLOWED_EXTENSIONS)):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Allowed types: {', '.join(settings.ALLOWED_EXTENSIONS)}"
            )
        
        # Initialize services
        visual_search_service = VisualSearchService()
        
        # Process image and get results
        results = await visual_search_service.search(
            image=image,
            limit=limit,
            min_confidence=min_confidence
        )
        
        return SearchResponse(
            success=True,
            results=results,
            message="Search completed successfully"
        )
    
    except Exception as e:
        logger.error(f"Error in visual search: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/outfit-recommendations", response_model=List[OutfitRecommendation])
async def get_outfit_recommendations(
    image: UploadFile = File(...),
    limit: int = 5,
    style_preference: Optional[str] = None
):
    """
    Get outfit recommendations based on uploaded item
    """
    try:
        # Initialize services
        outfit_service = OutfitRecommenderService()
        
        # Get recommendations
        recommendations = await outfit_service.get_recommendations(
            image=image,
            limit=limit,
            style_preference=style_preference
        )
        
        return recommendations
    
    except Exception as e:
        logger.error(f"Error in outfit recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/trends")
async def get_fashion_trends():
    """
    Get current fashion trends
    """
    try:
        # TODO: Implement trend analysis service
        return JSONResponse(
            content={
                "success": True,
                "trends": [
                    {"category": "Colors", "trends": ["sage green", "dusty rose"]},
                    {"category": "Patterns", "trends": ["floral", "geometric"]},
                    {"category": "Styles", "trends": ["oversized", "minimalist"]}
                ]
            }
        )
    except Exception as e:
        logger.error(f"Error fetching trends: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 