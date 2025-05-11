import os
import json
import pickle
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Union
import faiss
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class VectorStore:
    """Manages dog embeddings in a vector database for similarity search."""

    def __init__(self, embedding_dim: int = 1536, index_path: Optional[str] = None,
                 metadata_path: Optional[str] = None):
        """
        Initialize the vector store with optional existing index.

        Args:
            embedding_dim: Dimension of the embedding vectors
            index_path: Optional path to an existing FAISS index
            metadata_path: Optional path to metadata for the index
        """
        self.embedding_dim = embedding_dim
        self.index = None
        self.metadata = []

        # Create or load index
        if index_path and os.path.exists(index_path) and metadata_path and os.path.exists(metadata_path):
            self.load(index_path, metadata_path)
        else:
            self.index = faiss.IndexFlatL2(embedding_dim)
            self.metadata = []

    def add_embeddings(self, embeddings: List[List[float]], metadata_list: List[Dict[str, Any]]) -> None:
        """
        Add embeddings and their metadata to the vector store.

        Args:
            embeddings: List of embedding vectors
            metadata_list: List of metadata dictionaries corresponding to each embedding
        """
        if len(embeddings) != len(metadata_list):
            raise ValueError("Number of embeddings and metadata entries must match")

        if not embeddings:
            logger.warning("No embeddings to add")
            return

        # Convert embeddings to numpy array
        embeddings_np = np.array(embeddings).astype(np.float32)

        # Add to FAISS index
        self.index.add(embeddings_np)

        # Add metadata
        self.metadata.extend(metadata_list)

        logger.info(f"Added {len(embeddings)} embeddings to vector store. Total: {len(self.metadata)}")

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for the most similar embeddings.

        Args:
            query_embedding: Embedding vector to search for
            top_k: Number of results to return

        Returns:
            List of metadata dictionaries for the most similar embeddings
        """
        if not self.index:
            logger.warning("No index available for search")
            return []

        if len(self.metadata) == 0:
            logger.warning("No metadata available for search")
            return []

        # Convert query to numpy array
        query_np = np.array([query_embedding]).astype(np.float32)

        # Perform search
        distances, indices = self.index.search(query_np, min(top_k, len(self.metadata)))

        # Extract results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.metadata) and idx >= 0:  # Ensure index is valid
                result = self.metadata[idx].copy()
                result["similarity_score"] = float(
                    1.0 / (1.0 + distances[0][i]))  # Convert distance to similarity score
                results.append(result)

        return results

    def search_by_text(self, embedder, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search using a text query.

        Args:
            embedder: Embedder object to convert text to embedding
            query_text: Text query to search for
            top_k: Number of results to return

        Returns:
            List of metadata dictionaries for the most similar embeddings
        """
        # Get embedding for query text
        query_embedding = embedder.get_embedding(query_text)

        # Search using the embedding
        return self.search(query_embedding, top_k)

    def save(self, index_path: str, metadata_path: str) -> Tuple[str, str]:
        """
        Save the index and metadata to disk.

        Args:
            index_path: Path to save the FAISS index
            metadata_path: Path to save the metadata

        Returns:
            Tuple of paths where the index and metadata were saved
        """
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, index_path)

        # Save metadata
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)

        logger.info(f"Saved vector store to {index_path} and {metadata_path}")
        return index_path, metadata_path

    def load(self, index_path: str, metadata_path: str) -> bool:
        """
        Load the index and metadata from disk.

        Args:
            index_path: Path to the FAISS index
            metadata_path: Path to the metadata

        Returns:
            True if loading was successful, False otherwise
        """
        try:
            # Load FAISS index
            self.index = faiss.read_index(index_path)

            # Load metadata
            with open(metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)

            logger.info(f"Loaded vector store from {index_path} and {metadata_path}")
            logger.info(f"Vector store contains {len(self.metadata)} embeddings")
            return True

        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            # Create a new empty index
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            self.metadata = []
            return False

    def clear(self) -> None:
        """Clear the vector store."""
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.metadata = []
        logger.info("Vector store cleared")

    def get_size(self) -> int:
        """
        Get the number of embeddings in the vector store.

        Returns:
            Number of embeddings
        """
        return len(self.metadata)