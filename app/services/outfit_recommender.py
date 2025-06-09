import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Optional
import numpy as np
import logging
from datetime import datetime
import uuid

from app.core.config import settings
from app.schemas.search import OutfitRecommendation, OutfitItem, Product
from app.services.visual_search import VisualSearchService

logger = logging.getLogger(__name__)

class OutfitCompatibilityGNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.conv1 = nn.Linear(input_dim, hidden_dim)
        self.conv2 = nn.Linear(hidden_dim, hidden_dim)
        self.conv3 = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return torch.sigmoid(self.conv3(x))

class OutfitRecommenderService:
    def __init__(self):
        self.visual_search = VisualSearchService()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize GNN model
        self.model = OutfitCompatibilityGNN(
            input_dim=512,  # CLIP embedding dimension
            hidden_dim=256
        ).to(self.device)
        
        # Load pre-trained weights if available
        try:
            self.model.load_state_dict(torch.load("models/outfit_gnn.pth"))
            logger.info("Loaded pre-trained outfit compatibility model")
        except:
            logger.warning("No pre-trained model found, using untrained model")
    
    async def get_recommendations(
        self,
        image: bytes,
        limit: int = 5,
        style_preference: Optional[str] = None
    ) -> List[OutfitRecommendation]:
        """
        Generate outfit recommendations based on uploaded item
        """
        try:
            # Get similar items from visual search
            similar_items = await self.visual_search.search(
                image=image,
                limit=20,  # Get more items for better recommendations
                min_confidence=0.6
            )
            
            if not similar_items:
                raise ValueError("No similar items found")
            
            # Generate outfit recommendations
            recommendations = []
            for _ in range(limit):
                # Select base item
                base_item = similar_items[0]  # Use the most similar item
                
                # Generate compatible items
                compatible_items = self._generate_compatible_items(
                    base_item=base_item,
                    similar_items=similar_items,
                    style_preference=style_preference
                )
                
                # Create outfit recommendation
                outfit = OutfitRecommendation(
                    outfit_id=str(uuid.uuid4()),
                    items=compatible_items,
                    style=style_preference or "casual",
                    occasion=self._determine_occasion(compatible_items),
                    confidence_score=self._calculate_outfit_confidence(compatible_items),
                    explanation=self._generate_outfit_explanation(compatible_items),
                    created_at=datetime.utcnow()
                )
                
                recommendations.append(outfit)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating outfit recommendations: {str(e)}")
            raise
    
    def _generate_compatible_items(
        self,
        base_item: Product,
        similar_items: List[Product],
        style_preference: Optional[str]
    ) -> List[OutfitItem]:
        """
        Generate a list of compatible items for the base item
        """
        compatible_items = []
        
        # Add base item
        compatible_items.append(OutfitItem(
            product=base_item,
            match_reason="Base item",
            compatibility_score=1.0
        ))
        
        # Select complementary items
        for item in similar_items[1:]:
            if self._is_compatible(base_item, item, style_preference):
                compatibility_score = self._calculate_compatibility_score(base_item, item)
                if compatibility_score > 0.7:  # Only add highly compatible items
                    compatible_items.append(OutfitItem(
                        product=item,
                        match_reason=self._generate_match_reason(base_item, item),
                        compatibility_score=compatibility_score
                    ))
        
        return compatible_items
    
    def _is_compatible(
        self,
        item1: Product,
        item2: Product,
        style_preference: Optional[str]
    ) -> bool:
        """
        Check if two items are compatible
        """
        # TODO: Implement more sophisticated compatibility logic
        # For now, use simple category-based rules
        if item1.category == item2.category:
            return False
        
        if style_preference:
            return style_preference.lower() in item2.attributes.get("style", "").lower()
        
        return True
    
    def _calculate_compatibility_score(
        self,
        item1: Product,
        item2: Product
    ) -> float:
        """
        Calculate compatibility score between two items
        """
        # TODO: Implement more sophisticated scoring
        # For now, use a simple random score
        return np.random.uniform(0.7, 1.0)
    
    def _determine_occasion(self, items: List[OutfitItem]) -> str:
        """
        Determine the occasion for the outfit
        """
        # TODO: Implement occasion detection logic
        return "casual"
    
    def _calculate_outfit_confidence(self, items: List[OutfitItem]) -> float:
        """
        Calculate overall confidence score for the outfit
        """
        if not items:
            return 0.0
        
        # Average of individual compatibility scores
        return sum(item.compatibility_score for item in items) / len(items)
    
    def _generate_match_reason(self, base_item: Product, item: Product) -> str:
        """
        Generate a human-readable reason for why items match
        """
        # TODO: Implement more sophisticated explanation generation
        return f"Complements the {base_item.category} with matching style"
    
    def _generate_outfit_explanation(self, items: List[OutfitItem]) -> str:
        """
        Generate a human-readable explanation for the outfit
        """
        if not items:
            return "No items in outfit"
        
        item_descriptions = [
            f"{item.product.category} by {item.product.brand}"
            for item in items
        ]
        
        return f"This outfit combines {' and '.join(item_descriptions)} for a cohesive look." 