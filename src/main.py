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

    # Verify key environment variables (will be used later)
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        logger.info("OpenAI API key found")
    else:
        logger.warning("Warning: OpenAI API key not found. You'll need this for the RAG system.")

    # Import data collection module
    try:
        from data_collection import run_data_collection

        # Ask user if they want to run data collection
        run_collection = input("Do you want to run data collection? (y/n): ").lower() == 'y'

        if run_collection:
            logger.info("Starting data collection...")
            result = run_data_collection()

            # Log results
            if result["processed"]:
                logger.info(f"Data collection successful. Processed data saved to {result['processed']}")
            else:
                logger.warning("Data collection completed but no processed data was generated.")
        else:
            logger.info("Skipping data collection.")

    except ImportError:
        logger.error("Could not import data collection module. Make sure all dependencies are installed.")

    logger.info("Shelter Match RAG - Initialization Complete")


if __name__ == "__main__":
    main()