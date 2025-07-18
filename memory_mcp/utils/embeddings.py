"""
Embedding utilities for the memory MCP server.
"""

import os
from typing import Any, Dict, List, Optional, Union

import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer


class EmbeddingManager:
    """
    Manages embedding generation and similarity calculations.
    
    This class handles the loading of embedding models, generation
    of embeddings for text, and calculation of similarity between
    embeddings.
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the embedding manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.model_name = config["embedding"].get("model", "sentence-transformers/all-MiniLM-L6-v2")
        self.dimensions = config["embedding"].get("dimensions", 384)
        self.cache_dir = config["embedding"].get("cache_dir", None)
        
        # Model will be loaded on first use
        self.model = None
    
    def get_model(self) -> SentenceTransformer:
        """
        Get or load the embedding model.
        
        Returns:
            SentenceTransformer model
        """
        if self.model is None:
            # Create cache directory if specified
            if self.cache_dir:
                os.makedirs(self.cache_dir, exist_ok=True)
                
            # Load model
            logger.info(f"Loading embedding model: {self.model_name}")
            try:
                self.model = SentenceTransformer(
                    self.model_name,
                    cache_folder=self.cache_dir
                )
                logger.info(f"Embedding model loaded: {self.model_name}")
            except Exception as e:
                logger.error(f"Error loading embedding model: {str(e)}")
                raise RuntimeError(f"Failed to load embedding model: {str(e)}")
        
        return self.model
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate an embedding vector for text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as a list of floats
        """
        model = self.get_model()
        
        # Generate embedding
        try:
            embedding = model.encode(text)
            
            # Convert to list of floats for JSON serialization
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            # Return zero vector as fallback
            return [0.0] * self.dimensions
    
    def batch_generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        model = self.get_model()
        
        # Generate embeddings in batch
        try:
            embeddings = model.encode(texts)
            
            # Convert to list of lists for JSON serialization
            return [embedding.tolist() for embedding in embeddings]
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {str(e)}")
            # Return zero vectors as fallback
            return [[0.0] * self.dimensions] * len(texts)
    
    def calculate_similarity(
        self,
        embedding1: Union[List[float], np.ndarray],
        embedding2: Union[List[float], np.ndarray]
    ) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity (0.0-1.0)
        """
        # Convert to numpy arrays if needed
        if isinstance(embedding1, list):
            embedding1 = np.array(embedding1)
        if isinstance(embedding2, list):
            embedding2 = np.array(embedding2)
        
        # Calculate cosine similarity
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(np.dot(embedding1, embedding2) / (norm1 * norm2))
    
    def find_most_similar(
        self,
        query_embedding: Union[List[float], np.ndarray],
        embeddings: List[Union[List[float], np.ndarray]],
        min_similarity: float = 0.0,
        limit: int = 5
    ) -> List[Dict[str, Union[int, float]]]:
        """
        Find most similar embeddings to a query embedding.
        
        Args:
            query_embedding: Query embedding vector
            embeddings: List of embeddings to compare against
            min_similarity: Minimum similarity threshold
            limit: Maximum number of results
            
        Returns:
            List of dictionaries with index and similarity
        """
        # Convert query to numpy array if needed
        if isinstance(query_embedding, list):
            query_embedding = np.array(query_embedding)
        
        # Calculate similarities
        similarities = []
        
        for i, embedding in enumerate(embeddings):
            # Convert to numpy array if needed
            if isinstance(embedding, list):
                embedding = np.array(embedding)
            
            # Calculate similarity
            similarity = self.calculate_similarity(query_embedding, embedding)
            
            if similarity >= min_similarity:
                similarities.append({
                    "index": i,
                    "similarity": similarity
                })
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        
        # Limit results
        return similarities[:limit]
