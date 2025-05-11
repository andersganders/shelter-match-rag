import os
import sys
import logging
from dotenv import load_dotenv
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import our utility functions
try:
    from utils import load_environment
except ImportError:
    logger.error("Could not import utility functions.")
    load_dotenv()  # Fallback to simple dotenv loading


def main():
    print("\n=== Shelter Match RAG - Starting Up ===\n")

    # Load and validate environment variables
    try:
        env_vars = load_environment()

        # Check OpenAI API key
        api_key = env_vars.get("OPENAI_API_KEY")
        if api_key:
            logger.info("OpenAI API key found")
        else:
            logger.warning("OpenAI API key not found. You'll need this for the RAG system.")
            print("\nNo OpenAI API key found. You need an API key to use this system.")
            print("You can set up your API key by running: python setup.py")
            return
    except:
        # Fallback if utils module is not available
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            logger.info("OpenAI API key found")
        else:
            logger.warning("OpenAI API key not found. You'll need this for the RAG system.")
            print("\nNo OpenAI API key found. You need an API key to use this system.")
            print("You can set up your API key by creating a .env file with: OPENAI_API_KEY=your_key_here")
            return

    # Get user choice for what to run
    print("\nWhat would you like to do?")
    print("1. Run data collection")
    print("2. Process data and create embeddings")
    print("3. Test embedding with sample queries")
    print("4. Test questionnaire analysis")
    print("5. Run complete pipeline (collection + embeddings)")
    print("6. Exit")

    choice = input("Enter your choice (1-6): ")

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

            # Check if sample data exists and offer it as an alternative
            sample_path = "data/sample/sample_dogs.csv"
            if os.path.exists(sample_path) and not os.path.exists(default_input):
                default_input = sample_path

            input_path = input(f"Enter path to dog data CSV (default: {default_input}): ") or default_input

            if not os.path.exists(input_path):
                logger.error(f"Input file {input_path} does not exist.")

                # Offer to create sample data
                create_sample = input("Would you like to create sample data for testing? (y/n): ").lower() == 'y'

                if create_sample:
                    # Create sample directory if it doesn't exist
                    os.makedirs("data/sample", exist_ok=True)

                    # Create a Python script to generate sample data
                    from embedding.run_sample_embedding import create_sample_dog_data
                    input_path, _ = create_sample_dog_data()
                else:
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

                while True:
                    # Get user query
                    query = input("\nEnter a query describing the dog you're looking for (or 'exit' to quit): ")

                    if query.lower() in ['exit', 'quit', 'q']:
                        break

                    if query:
                        # Run query
                        results = pipeline.search_similar_dogs(query, top_k=3)

                        # Display results
                        print("\nTop matches:")
                        for i, result in enumerate(results, 1):
                            match_score = int(result['similarity_score'] * 100)
                            print(f"{i}. {result['name']} ({result['breed']}) - Match: {match_score}%")
                            print(f"   Sex: {result['sex']}, Age: {result['age']}, Weight: {result['weight']}")
                            print(f"   Description: {result.get('description_snippet', 'N/A')}")
                            print("")
                    else:
                        logger.warning("No query provided.")
            else:
                logger.error("Failed to load vector store. Run option 2 first to create embeddings.")

                # Offer to create sample data and embeddings
                create_sample = input(
                    "Would you like to create sample data and embeddings for testing? (y/n): ").lower() == 'y'

                if create_sample:
                    # Create sample directory if it doesn't exist
                    os.makedirs("data/sample", exist_ok=True)

                    # Create sample data and embeddings
                    from embedding.run_sample_embedding import create_sample_dog_data
                    sample_path, _ = create_sample_dog_data()

                    # Create embeddings
                    pipeline = EmbeddingPipeline()
                    logger.info(f"Processing sample dog data from {sample_path}...")
                    index_path, metadata_path = pipeline.process_dog_data(sample_path)

                    if index_path and metadata_path:
                        logger.info(f"Embedding successful. Vector store saved to {index_path} and {metadata_path}")

                        # Try loading again
                        if pipeline.load_latest_vector_store():
                            logger.info("Loaded vector store successfully")

                            # Offer to run a test query
                            test_query = input("\nEnter a test query (or press Enter to skip): ")
                            if test_query:
                                results = pipeline.search_similar_dogs(test_query, top_k=3)

                                # Display results
                                print("\nTop matches:")
                                for i, result in enumerate(results, 1):
                                    match_score = int(result['similarity_score'] * 100)
                                    print(f"{i}. {result['name']} ({result['breed']}) - Match: {match_score}%")
                                    print(f"   Sex: {result['sex']}, Age: {result['age']}, Weight: {result['weight']}")
                                    print(f"   Description: {result.get('description_snippet', 'N/A')}")
                                    print("")
                    else:
                        logger.warning("Embedding sample data failed.")

        except ImportError:
            logger.error("Could not import embedding module. Make sure all dependencies are installed.")

    elif choice == "4":
        # Test questionnaire analysis
        try:
            # Run the test_questionnaire.py script
            logger.info("Running questionnaire analysis test...")

            try:
                from matching.test_questionnaire import test_questionnaire_analysis
                test_questionnaire_analysis()
            except ImportError:
                logger.error("Could not import test_questionnaire module.")
                print("Make sure all dependencies are installed and the module is in the correct location.")

        except Exception as e:
            logger.error(f"Error running questionnaire analysis test: {e}")

    elif choice == "5":
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

                    # Ask if user wants to test a query
                    test_query = input("\nDo you want to test a query? (y/n): ").lower() == 'y'

                    if test_query:
                        # Load the vector store
                        if pipeline.load_latest_vector_store():
                            query = input("Enter your query: ")

                            if query:
                                # Run query
                                results = pipeline.search_similar_dogs(query, top_k=3)

                                # Display results
                                print("\nTop matches:")
                                for i, result in enumerate(results, 1):
                                    match_score = int(result['similarity_score'] * 100)
                                    print(f"{i}. {result['name']} ({result['breed']}) - Match: {match_score}%")
                                    print(f"   Sex: {result['sex']}, Age: {result['age']}, Weight: {result['weight']}")
                                    print(f"   Description: {result.get('description_snippet', 'N/A')}")
                                    print("")
                else:
                    logger.warning("Embedding completed but no vector store was generated.")
            else:
                logger.warning("Data collection completed but no processed data was generated.")

                # Offer to use sample data instead
                use_sample = input("Would you like to use sample data instead? (y/n): ").lower() == 'y'

                if use_sample:
                    # Create sample data
                    from embedding.run_sample_embedding import create_sample_dog_data
                    sample_path, _ = create_sample_dog_data()

                    # Create embeddings
                    pipeline = EmbeddingPipeline()
                    logger.info(f"Processing sample dog data from {sample_path}...")
                    index_path, metadata_path = pipeline.process_dog_data(sample_path)

                    if index_path and metadata_path:
                        logger.info(f"Embedding successful. Vector store saved to {index_path} and {metadata_path}")
        except ImportError as e:
            logger.error(f"Could not import required modules: {e}")

    elif choice == "6":
        logger.info("Exiting...")
        return

    else:
        logger.warning("Invalid choice. Please enter a number between 1 and 6.")

    logger.info("Shelter Match RAG - Operation Complete")


if __name__ == "__main__":
    main()
