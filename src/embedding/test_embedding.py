import os
import pandas as pd
import logging
from dotenv import load_dotenv
from typing import List, Dict, Any

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import our embedding modules
from embedder import Embedder
from vector_store import VectorStore
from embedding_pipeline import EmbeddingPipeline


def create_sample_dog_data(output_path: str = "data/sample/sample_dogs.csv") -> str:
    """
    Create a small sample dataset for testing embeddings.

    Args:
        output_path: Path to save the sample CSV

    Returns:
        Path to the saved file
    """
    # Create sample data
    sample_data = [
        {
            "dog_id": "1",
            "name": "Buddy",
            "breed": "Labrador Retriever",
            "sex": "Male",
            "age": "3 years",
            "weight": "70 lbs",
            "description": "Buddy is a friendly and energetic Labrador who loves to play fetch and swim. He's good with children and other dogs. He's house-trained and knows basic commands like sit, stay, and come.",
            "data_source": "sample",
            "source_id": "sample_1"
        },
        {
            "dog_id": "2",
            "name": "Daisy",
            "breed": "Beagle",
            "sex": "Female",
            "age": "2 years",
            "weight": "25 lbs",
            "description": "Daisy is a sweet Beagle who loves cuddles and short walks. She's a bit shy at first but warms up quickly. She's good with other dogs but hasn't been tested with cats. She would do best in a quieter home without small children.",
            "data_source": "sample",
            "source_id": "sample_2"
        },
        {
            "dog_id": "3",
            "name": "Max",
            "breed": "German Shepherd",
            "sex": "Male",
            "age": "5 years",
            "weight": "85 lbs",
            "description": "Max is a loyal and intelligent German Shepherd. He has received some training and responds well to commands. He's protective of his family and would do best in a home with an experienced dog owner who can provide structure and exercise.",
            "data_source": "sample",
            "source_id": "sample_3"
        },
        {
            "dog_id": "4",
            "name": "Bella",
            "breed": "Chihuahua mix",
            "sex": "Female",
            "age": "8 years",
            "weight": "8 lbs",
            "description": "Bella is a senior Chihuahua mix who enjoys short walks and lots of lap time. She's very loving with her people but can be wary of strangers. She would do best as the only pet in a quiet adult home.",
            "data_source": "sample",
            "source_id": "sample_4"
        },
        {
            "dog_id": "5",
            "name": "Rocky",
            "breed": "Pit Bull Terrier",
            "sex": "Male",
            "age": "4 years",
            "weight": "55 lbs",
            "description": "Rocky is a playful and affectionate Pit Bull Terrier. He loves people of all ages and gets along well with most other dogs. He has a lot of energy and would thrive in an active home that can provide plenty of exercise and mental stimulation.",
            "data_source": "sample",
            "source_id": "sample_5"
        }
    ]

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Create DataFrame and save to CSV
    df = pd.DataFrame(sample_data)
    df.to_csv(output_path, index=False)

    logger.info(f"Created sample dog data with {len(df)} records at {output_path}")
    return output_path


def test_embeddings() -> None:
    """Test the embedding pipeline with sample data."""
    # Create sample data
    sample_path = create_sample_dog_data()

    # Initialize embedding pipeline
    pipeline = EmbeddingPipeline()

    # Process sample data
    logger.info("Processing sample dog data...")
    index_path, metadata_path = pipeline.process_dog_data(sample_path)

    if not index_path or not metadata_path:
        logger.error("Failed to process sample data")
        return

    logger.info(f"Created vector store at {index_path} and {metadata_path}")

    # Test queries
    test_queries = [
        "I'm looking for a large, energetic dog that's good with kids",
        "I want a small, quiet dog that doesn't need much exercise",
        "I need a well-trained dog that can protect my home",
        "I'd like a dog that's good for apartments and doesn't bark much",
        "I want a friendly dog that gets along with other dogs"
    ]

    # Load latest vector store
    if pipeline.load_latest_vector_store():
        logger.info("Loaded vector store successfully")

        # Run test queries
        for i, query in enumerate(test_queries, 1):
            logger.info(f"\nTest Query #{i}: '{query}'")
            results = pipeline.search_similar_dogs(query, top_k=2)

            for j, result in enumerate(results, 1):
                logger.info(
                    f"Match #{j}: {result['name']} ({result['breed']}) - Score: {result['similarity_score']:.4f}")
                logger.info(f"  Description: {result.get('description_snippet', 'N/A')}")
    else:
        logger.error("Failed to load vector store")


if __name__ == "__main__":
    test_embeddings()