import os
import logging
from typing import List, Dict, Any, Optional, Union
import numpy as np
import pandas as pd
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Embedder:
    """Creates embeddings for text using different models."""

    def __init__(self, model_name: str = "text-embedding-ada-002"):
        """
        Initialize the embedder with a specific model.

        Args:
            model_name: Name of the embedding model to use
        """
        self.model_name = model_name

        # Set up OpenAI API key
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            logger.warning("OpenAI API key not found. Embeddings will not work without it.")
        else:
            openai.api_key = openai_api_key

    def _create_dog_profile_text(self, dog_data: Dict[str, Any]) -> str:
        """
        Create a standardized text representation of a dog's profile for embedding.

        Args:
            dog_data: Dictionary containing dog information

        Returns:
            Text representation of the dog's profile
        """
        # Create a consistent text format for embedding
        profile_parts = []

        # Add name
        name = dog_data.get('name', 'Unknown')
        profile_parts.append(f"Name: {name}")

        # Add breed
        breed = dog_data.get('breed', 'Unknown')
        profile_parts.append(f"Breed: {breed}")

        # Add sex
        sex = dog_data.get('sex', 'Unknown')
        profile_parts.append(f"Sex: {sex}")

        # Add age
        age = dog_data.get('age', 'Unknown')
        profile_parts.append(f"Age: {age}")

        # Add weight
        weight = dog_data.get('weight', 'Unknown')
        profile_parts.append(f"Weight: {weight}")

        # Add description
        description = dog_data.get('description', '')
        if description:
            profile_parts.append(f"Description: {description}")

        # Add any temperament information
        temperament = dog_data.get('temperament', '')
        if temperament:
            profile_parts.append(f"Temperament: {temperament}")

        # Add any special needs
        special_needs = dog_data.get('special_needs', '')
        if special_needs:
            profile_parts.append(f"Special Needs: {special_needs}")

        # Add any training information
        training = dog_data.get('training', '')
        if training:
            profile_parts.append(f"Training: {training}")

        # Combine all parts into a single text
        profile_text = "\n".join(profile_parts)
        return profile_text

    def get_embedding(self, text: str) -> List[float]:
        """
        Get embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as a list of floats
        """
        try:
            response = openai.Embedding.create(
                input=text,
                model=self.model_name
            )
            embedding = response['data'][0]['embedding']
            return embedding
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            # Return a zero vector as fallback (not ideal but prevents crashes)
            # This should be handled better in production
            return [0.0] * 1536  # Default size for OpenAI embeddings

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        embeddings = []

        # Process in batches to avoid rate limits
        batch_size = 20  # Adjust based on API limits
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                response = openai.Embedding.create(
                    input=batch,
                    model=self.model_name
                )
                batch_embeddings = [data['embedding'] for data in response['data']]
                embeddings.extend(batch_embeddings)

                # Log progress for large batches
                if len(texts) > batch_size:
                    logger.info(f"Embedded batch {i // batch_size + 1}/{(len(texts) - 1) // batch_size + 1}")

            except Exception as e:
                logger.error(f"Error getting embeddings for batch {i // batch_size + 1}: {e}")
                # Add zero vectors as fallback
                zero_vector = [0.0] * 1536  # Default size for OpenAI embeddings
                embeddings.extend([zero_vector] * len(batch))

        return embeddings

    def embed_dogs_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create embeddings for all dogs in a DataFrame.

        Args:
            df: DataFrame containing dog data

        Returns:
            DataFrame with added embedding column
        """
        # Create a copy to avoid modifying the original
        result_df = df.copy()

        # Create text profiles for each dog
        profiles = []
        for _, row in df.iterrows():
            profile_text = self._create_dog_profile_text(row.to_dict())
            profiles.append(profile_text)

        # Get embeddings for all profiles
        logger.info(f"Creating embeddings for {len(profiles)} dog profiles...")
        embeddings = self.get_embeddings(profiles)

        # Add embeddings and profiles to the DataFrame
        result_df['profile_text'] = profiles
        result_df['embedding'] = embeddings

        logger.info(f"Created embeddings for {len(embeddings)} dog profiles")
        return result_df