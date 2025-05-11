import os
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    print("Shelter Match RAG - Starting Up")

    # Verify key environment variables
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        logger.info("OpenAI API key found")
    else:
        logger.warning("Warning: OpenAI API key not found. You'll need this for the RAG system.")

    # Get user choice for what to run
    print("\nWhat would you like to do?")
    print("1. Run data collection")
    print("2. Process data and create embeddings")
    print("3. Test embedding with sample queries")
    print("4. Run complete pipeline (collection + embeddings)")
    print("5. Exit")

    choice = input("Enter your choice (1-5): ")

    if choice == "1":
        # Import and run data collection
        try:
            from data_collection import run_data_collection

            logger.info("Starting data collection...")
            result = run_data_collection()

            # Log results
            if result["processed"]:
                logger.info(f"Data collection successful. Processed data saved to {result['processed']}")
            else:
                logger.warning("Data collection completed but no processed data was generated.")

        except ImportError:
            logger.error("Could not import data collection module. Make sure all dependencies are installed.")

    elif choice == "2":
        # Import and run embedding pipeline
        try:
            from embedding import EmbeddingPipeline

            # Ask for input file
            default_input = "data/processed/all_dogs_latest.csv"
            input_path = input(f"Enter path to dog data CSV (default: {default_input}): ") or default_input

            if not os.path.exists(input_path):
                logger.error(f"Input file {input_path} does not exist.")
                return

            # Initialize and run pipeline
            pipeline = EmbeddingPipeline()
            logger.info(f"Processing dog data from {input_path}...")
            index_path, metadata_path = pipeline.process_dog_data(input_path)

            if index_path and metadata_path:
                logger.info(f"Embedding successful. Vector store saved to {index_path} and {metadata_path}")
            else:
                logger.warning("Embedding completed but no vector store was generated.")

        except ImportError:
            logger.error("Could not import embedding module. Make sure all dependencies are installed.")

    elif choice == "3":
        # Test embedding with sample queries
        try:
            from embedding import EmbeddingPipeline

            # Initialize pipeline
            pipeline = EmbeddingPipeline()

            # Load latest vector store
            if pipeline.load_latest_vector_store():
                logger.info("Loaded vector store successfully")

                # Get user query
                query = input("\nEnter a query describing the dog you're looking for: ")

                if query:
                    # Run query
                    results = pipeline.search_similar_dogs(query, top_k=3)

                    # Display results
                    print("\nTop matches:")
                    for i, result in enumerate(results, 1):
                        print(f"{i}. {result['name']} ({result['breed']}) - Score: {result['similarity_score']:.4f}")
                        print(f"   Description: {result.get('description_snippet', 'N/A')}")
                        print("")
                else:
                    logger.warning("No query provided.")
            else:
                logger.error("Failed to load vector store. Run option 2 first to create embeddings.")

        except ImportError:
            logger.error("Could not import embedding module. Make sure all dependencies are installed.")

    elif choice == "4":
        # Run complete pipeline
        try:
            from data_collection import run_data_collection
            from embedding import EmbeddingPipeline

            # Run data collection
            logger.info("Starting data collection...")
            result = run_data_collection()

            if result["processed"]:
                logger.info(f"Data collection successful. Processed data saved to {result['processed']}")

                # Run embedding pipeline
                pipeline = EmbeddingPipeline()
                logger.info(f"Processing dog data from {result['processed']}...")
                index_path, metadata_path = pipeline.process_dog_data(result["processed"])

                if index_path and metadata_path:
                    logger.info(f"Embedding successful. Vector store saved to {index_path} and {metadata_path}")
                else:
                    logger.warning("Embedding completed but no vector store was generated.")
            else:
                logger.warning("Data collection completed but no processed data was generated.")

        except ImportError as e:
            logger.error(f"Could not import required modules: {e}")

    elif choice == "5":
        logger.info("Exiting...")
        return

    else:
        logger.warning("Invalid choice. Please enter a number between 1 and 5.")

    logger.info("Shelter Match RAG - Operation Complete")


if __name__ == "__main__":
    main()