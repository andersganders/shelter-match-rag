import os
import sys
import pandas as pd
import logging
from dotenv import load_dotenv
from pathlib import Path

# Add parent directory to path to allow importing from sibling packages
sys.path.append(str(Path(__file__).parent.parent))

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import embedding modules
from embedding.embedder import Embedder
from embedding.vector_store import VectorStore
from embedding.embedding_pipeline import EmbeddingPipeline


def create_sample_dog_data():
    """Create a small sample dataset for testing embeddings."""
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
    os.makedirs("data/sample", exist_ok=True)

    # Create DataFrame and save to CSV
    output_path = "data/sample/sample_dogs.csv"
    df = pd.DataFrame(sample_data)
    df.to_csv(output_path, index=False)

    logger.info(f"Created sample dog data with {len(df)} records at {output_path}")
    return output_path, df


def run_sample_embedding():
    """Run the embedding pipeline on sample data and test queries."""
    print("\n=== Shelter Match RAG - Embedding Sample Runner ===\n")

    # Check for OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️ OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
        return

    # Create sample data
    print("Creating sample dog data...")
    sample_path, sample_df = create_sample_dog_data()

    # Initialize embedding pipeline
    pipeline = EmbeddingPipeline()

    # Process sample data
    print("\nProcessing sample dog data and creating embeddings...")
    index_path, metadata_path = pipeline.process_dog_data(sample_path)

    if not index_path or not metadata_path:
        print("❌ Failed to process sample data")
        return

    print(f"✅ Created vector store at {index_path}")

    # Load the vector store we just created
    if pipeline.load_latest_vector_store():
        print("✅ Loaded vector store successfully")

        # Display sample dogs
        print("\n=== Sample Dogs in Database ===")
        for i, row in sample_df.iterrows():
            print(f"{i + 1}. {row['name']} - {row['breed']} ({row['sex']}, {row['age']})")

        # Interactive query loop
        print("\n=== Test the Retrieval System ===")
        print("Enter queries to test the embedding retrieval system.")
        print("Examples:")
        print("- I'm looking for a large dog that's good with kids")
        print("- I need a small, quiet dog for an apartment")
        print("- I want an active dog that can join me on runs")
        print("- I'd like a well-trained, intelligent dog")
        print("(Type 'exit' to quit)")

        while True:
            print("\n-----------------------------------------------------")
            query = input("Enter your query: ")

            if query.lower() in ['exit', 'quit', 'q']:
                break

            if not query:
                continue

            # Run query
            print("\nSearching for matches...")
            results = pipeline.search_similar_dogs(query, top_k=3)

            # Display results
            if results:
                print("\n🔍 Top matches:")
                for i, result in enumerate(results, 1):
                    score_percentage = int(result['similarity_score'] * 100)
                    print(f"{i}. {result['name']} ({result['breed']}) - Match: {score_percentage}%")
                    print(f"   Sex: {result['sex']}, Age: {result['age']}, Weight: {result['weight']}")
                    print(f"   Description: {result.get('description_snippet', 'N/A')}")
                    print("")
            else:
                print("No matches found.")
    else:
        print("❌ Failed to load vector store")


if __name__ == "__main__":
    run_sample_embedding()