import torch
from transformers import CLIPProcessor, CLIPModel
import faiss
import numpy as np
from PIL import Image
import io
import logging
from typing import List, Dict, Any
import time

from app.core.config import settings
from app.schemas.search import Product

logger = logging.getLogger(__name__)

class VisualSearchService:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Initialize CLIP model
        self.model = CLIPModel.from_pretrained(settings.CLIP_MODEL_NAME).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(settings.CLIP_MODEL_NAME)
        
        # Initialize FAISS index
        self.index = self._load_faiss_index()
        
        # Load product metadata
        self.product_metadata = self._load_product_metadata()
    
    def _load_faiss_index(self) -> faiss.Index:
        """Load or create FAISS index"""
        try:
            index = faiss.read_index(settings.FAISS_INDEX_PATH)
            logger.info("Loaded existing FAISS index")
        except:
            # Create a new index if none exists
            dimension = 512  # CLIP embedding dimension
            index = faiss.IndexFlatL2(dimension)
            logger.info("Created new FAISS index")
        return index
    
    def _load_product_metadata(self) -> Dict[str, Any]:
        """Load product metadata from database"""
        # TODO: Implement database connection and metadata loading
        return {}
    
    async def search(
        self,
        image: bytes,
        limit: int = 10,
        min_confidence: float = 0.7
    ) -> List[Product]:
        """
        Perform visual search on uploaded image
        """
        start_time = time.time()
        
        try:
            # Process image
            image = Image.open(io.BytesIO(image))
            inputs = self.processor(
                images=image,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            # Get image embedding
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
                image_features = image_features / image_features.norm(dim=1, keepdim=True)
            
            # Convert to numpy for FAISS
            query_vector = image_features.cpu().numpy().astype('float32')
            
            # Search in FAISS index
            distances, indices = self.index.search(query_vector, limit)
            
            # Process results
            results = []
            for distance, idx in zip(distances[0], indices[0]):
                if idx == -1:  # FAISS returns -1 for empty slots
                    continue
                    
                confidence = 1.0 - (distance / 2.0)  # Convert distance to confidence
                if confidence < min_confidence:
                    continue
                
                # Get product metadata
                product_data = self.product_metadata.get(str(idx), {})
                
                # Create Product object
                product = Product(
                    id=str(idx),
                    name=product_data.get("name", "Unknown"),
                    brand=product_data.get("brand", "Unknown"),
                    price=product_data.get("price", 0.0),
                    image_url=product_data.get("image_url", ""),
                    category=product_data.get("category", "Unknown"),
                    confidence_score=float(confidence),
                    attributes=product_data.get("attributes", {})
                )
                results.append(product)
            
            processing_time = time.time() - start_time
            logger.info(f"Search completed in {processing_time:.2f} seconds")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in visual search: {str(e)}")
            raise
    
    async def add_to_index(self, image: bytes, product_data: Dict[str, Any]) -> bool:
        """
        Add a new product to the search index
        """
        try:
            # Process image
            image = Image.open(io.BytesIO(image))
            inputs = self.processor(
                images=image,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            # Get image embedding
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
                image_features = image_features / image_features.norm(dim=1, keepdim=True)
            
            # Convert to numpy for FAISS
            vector = image_features.cpu().numpy().astype('float32')
            
            # Add to FAISS index
            self.index.add(vector)
            
            # Save product metadata
            product_id = str(self.index.ntotal - 1)
            self.product_metadata[product_id] = product_data
            
            # Save updated index
            faiss.write_index(self.index, settings.FAISS_INDEX_PATH)
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding product to index: {str(e)}")
            return False 