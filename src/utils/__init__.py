import os
import logging
from typing import Dict
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_environment() -> Dict[str, str]:
    """
    Load and validate environment variables.

    Returns:
        Dictionary with loaded environment variables
    """
    # Load .env file if it exists
    load_dotenv()

    # Required environment variables
    required_vars = ["OPENAI_API_KEY"]

    # Optional environment variables with defaults
    optional_vars = {
        "PETPOINT_API_URL": "https://api.petpoint.com/api/v1",
        "PETPOINT_API_KEY": "",
        "RESCUEGROUPS_API_URL": "https://api.rescuegroups.org/v5",
        "RESCUEGROUPS_API_KEY": ""
    }

    # Check required variables
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)

    if missing_vars:
        logger.warning(f"Missing required environment variables: {', '.join(missing_vars)}")
        logger.warning("Please set these variables in a .env file or in your environment.")

    # Set defaults for optional variables if not present
    for var, default in optional_vars.items():
        if not os.getenv(var):
            os.environ[var] = default
            logger.info(f"Using default value for {var}: {default}")

    # Return all environment variables as a dictionary
    result = {}
    for var in required_vars + list(optional_vars.keys()):
        result[var] = os.getenv(var, "")

    return result


def create_dotenv_file(api_key: str = None) -> bool:
    """
    Create a .env file with the provided API key.

    Args:
        api_key: OpenAI API key

    Returns:
        True if successful, False otherwise
    """
    if not api_key:
        logger.warning("No API key provided.")
        return False

    try:
        with open(".env", "w") as f:
            f.write(f"OPENAI_API_KEY={api_key}\n")
            f.write("PETPOINT_API_URL=https://api.petpoint.com/api/v1\n")
            f.write("PETPOINT_API_KEY=\n")
            f.write("RESCUEGROUPS_API_URL=https://api.rescuegroups.org/v5\n")
            f.write("RESCUEGROUPS_API_KEY=\n")

        logger.info("Created .env file successfully.")
        return True
    except Exception as e:
        logger.error(f"Error creating .env file: {e}")
        return False