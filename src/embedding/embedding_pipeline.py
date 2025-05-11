import os
import logging
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from .embedder import Embedder
from .vector_store import VectorStore

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EmbeddingPipeline:
    """Pipeline for creating and managing embeddings for dog profiles."""

    def __init__(self,
                 model_name: str = "text-embedding-ada-002",
                 embedding_dim: int = 1536,
                 vector_store_dir: str = "data/vector_store"):
        """
        Initialize the embedding pipeline.

        Args:
            model_name: Name of the embedding model to use
            embedding_dim: Dimension of the embedding vectors
            vector_store_dir: Directory to store vector indices
        """
        self.embedder = Embedder(model_name)
        self.vector_store = VectorStore(embedding_dim)
        self.vector_store_dir = vector_store_dir

        # Ensure vector store directory exists
        os.makedirs(vector_store_dir, exist_ok=True)

    def _extract_metadata(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Extract metadata from DataFrame rows.

        Args:
            df: DataFrame containing dog data

        Returns:
            List of metadata dictionaries
        """
        metadata_list = []

        for _, row in df.iterrows():
            metadata = {
                "dog_id": row.get("dog_id", ""),
                "name": row.get("name", "Unknown"),
                "breed": row.get("breed", "Unknown"),
                "sex": row.get("sex", "Unknown"),
                "age": row.get("age", "Unknown"),
                "weight": row.get("weight", "Unknown"),
                "data_source": row.get("data_source", "Unknown"),
                "source_id": row.get("source_id", ""),
                "profile_text": row.get("profile_text", "")
            }

            # Include description if available (truncated for metadata)
            description = row.get("description", "")
            if description:
                metadata["description_snippet"] = description[:200] + "..." if len(description) > 200 else description

            metadata_list.append(metadata)

        return metadata_list

    def process_dog_data(self, input_path: str) -> Tuple[str, str]:
        """
        Process dog data CSV, create embeddings, and build vector store.

        Args:
            input_path: Path to the CSV file with dog data

        Returns:
            Tuple of paths to saved index and metadata
        """
        # Create timestamp for this run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Define output paths
        index_path = os.path.join(self.vector_store_dir, f"dog_index_{timestamp}.faiss")
        metadata_path = os.path.join(self.vector_store_dir, f"dog_metadata_{timestamp}.pkl")

        # Read dog data
        try:
            df = pd.read_csv(input_path)
            logger.info(f"Read {len(df)} dog records from {input_path}")
        except Exception as e:
            logger.error(f"Error reading dog data from {input_path}: {e}")
            return "", ""

        # Create embeddings
        df_with_embeddings = self.embedder.embed_dogs_dataframe(df)

        # Extract embeddings and metadata
        embeddings = df_with_embeddings['embedding'].tolist()
        metadata_list = self._extract_metadata(df_with_embeddings)

        # Add to vector store
        self.vector_store.add_embeddings(embeddings, metadata_list)

        # Save vector store
        saved_index_path, saved_metadata_path = self.vector_store.save(index_path, metadata_path)

        # Create latest symlinks
        latest_index_path = os.path.join(self.vector_store_dir, "dog_index_latest.faiss")
        latest_metadata_path = os.path.join(self.vector_store_dir, "dog_metadata_latest.pkl")

        # On Windows, symlinks are tricky, so just copy the files
        import shutil
        try:
            shutil.copy2(saved_index_path, latest_index_path)
            shutil.copy2(saved_metadata_path, latest_metadata_path)
            logger.info(f"Created copies of latest vector store index and metadata")
        except Exception as e:
            logger.error(f"Error creating latest copies: {e}")

        return saved_index_path, saved_metadata_path

    def load_latest_vector_store(self) -> bool:
        """
        Load the latest vector store.

        Returns:
            True if loading was successful, False otherwise
        """
        latest_index_path = os.path.join(self.vector_store_dir, "dog_index_latest.faiss")
        latest_metadata_path = os.path.join(self.vector_store_dir, "dog_metadata_latest.pkl")

        if os.path.exists(latest_index_path) and os.path.exists(latest_metadata_path):
            return self.vector_store.load(latest_index_path, latest_metadata_path)
        else:
            logger.warning("No latest vector store found")
            return False

    def search_similar_dogs(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for dogs similar to a query.

        Args:
            query_text: Text query describing preferences
            top_k: Number of results to return

        Returns:
            List of metadata dictionaries for the most similar dogs
        """
        return self.vector_store.search_by_text(self.embedder, query_text, top_k)