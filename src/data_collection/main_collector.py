import os
import logging
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import our data collection classes
from .petpoint_client import PetPointClient
from .rescuegroups_client import RescueGroupsClient
from .message_board_scraper import MessageBoardScraper
from .data_processor import DataProcessor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_data_collection():
    """Run the complete data collection and processing pipeline."""
    start_time = datetime.now()
    logger.info(f"Starting data collection at {start_time}")

    # Create timestamp for this run
    timestamp = start_time.strftime("%Y%m%d_%H%M%S")

    # Define output paths
    data_dir = "data"
    raw_dir = os.path.join(data_dir, "raw", timestamp)
    processed_dir = os.path.join(data_dir, "processed")

    # Create directories
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    # Initialize clients
    petpoint_client = PetPointClient()
    rescuegroups_client = RescueGroupsClient()
    message_board_scraper = MessageBoardScraper()
    data_processor = DataProcessor()

    # Define file paths for this run
    petpoint_path = os.path.join(raw_dir, "petpoint_dogs.csv")
    rescuegroups_path = os.path.join(raw_dir, "rescuegroups_dogs.csv")
    message_boards_path = os.path.join(raw_dir, "message_board_posts.csv")
    processed_path = os.path.join(processed_dir, f"all_dogs_{timestamp}.csv")
    latest_path = os.path.join(processed_dir, "all_dogs_latest.csv")

    # Collect data from each source
    logger.info("Collecting data from PetPoint...")
    petpoint_file = petpoint_client.fetch_and_save_dogs(petpoint_path)

    logger.info("Collecting data from RescueGroups...")
    rescuegroups_file = rescuegroups_client.fetch_and_save_dogs(rescuegroups_path)

    logger.info("Scraping message boards...")
    message_boards_file = message_board_scraper.scrape_all_message_boards(message_boards_path)

    # Process and merge all data
    logger.info("Processing and merging data...")
    processed_file = data_processor.merge_and_process_data(
        petpoint_path, rescuegroups_path, message_boards_path, processed_path)

    # Create a copy as "latest" for easy reference
    if processed_file:
        import shutil
        shutil.copy(processed_file, latest_path)
        logger.info(f"Copied processed data to {latest_path}")

    # Log completion
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    logger.info(f"Data collection completed in {duration:.2f} seconds")

    # Return the paths to the collected data
    return {
        "petpoint": petpoint_file,
        "rescuegroups": rescuegroups_file,
        "message_boards": message_boards_file,
        "processed": processed_file,
        "latest": latest_path if processed_file else ""
    }


if __name__ == "__main__":
    run_data_collection()