from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class Product(BaseModel):
    id: str
    name: str
    brand: str
    price: float
    currency: str = "USD"
    image_url: str
    category: str
    confidence_score: float
    attributes: Dict[str, Any] = Field(default_factory=dict)

class SearchResponse(BaseModel):
    success: bool
    results: List[Product]
    message: str
    processing_time: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class OutfitItem(BaseModel):
    product: Product
    match_reason: str
    compatibility_score: float

class OutfitRecommendation(BaseModel):
    outfit_id: str
    items: List[OutfitItem]
    style: str
    occasion: Optional[str]
    confidence_score: float
    explanation: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

class TrendCategory(BaseModel):
    category: str
    trends: List[str]
    confidence: float
    source: str
    last_updated: datetime

class TrendResponse(BaseModel):
    success: bool
    categories: List[TrendCategory]
    timestamp: datetime = Field(default_factory=datetime.utcnow) 