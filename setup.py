import os
import sys
import subprocess
import logging
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import our utility functions
try:
    from utils import create_dotenv_file
except ImportError:
    logger.error("Could not import utility functions. Make sure you're running this script from the project root.")
    sys.exit(1)


def check_dependencies():
    """Check if all required dependencies are installed."""
    required_packages = [
        "openai>=1.0.0",
        "langchain",
        "faiss-cpu",
        "pandas",
        "numpy",
        "requests",
        "beautifulsoup4",
        "python-dotenv",
        "streamlit"
    ]

    missing_packages = []

    for package in required_packages:
        package_name = package.split('>=')[0]
        try:
            __import__(package_name)
        except ImportError:
            missing_packages.append(package)

    return missing_packages


def install_dependencies(missing_packages):
    """Install missing dependencies."""
    if not missing_packages:
        logger.info("All dependencies are already installed.")
        return True

    logger.info(f"Installing missing dependencies: {', '.join(missing_packages)}")

    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_packages)
        logger.info("Dependencies installed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error installing dependencies: {e}")
        return False


def setup_environment():
    """Set up the environment for the project."""
    print("\n=== Shelter Match RAG - Setup ===\n")

    # Check and install dependencies
    print("Checking dependencies...")
    missing_packages = check_dependencies()

    if missing_packages:
        print(f"The following packages need to be installed: {', '.join(missing_packages)}")
        install = input("Do you want to install them now? (y/n): ").lower() == 'y'

        if install:
            success = install_dependencies(missing_packages)
            if not success:
                print("There was an error installing dependencies. Please try installing them manually.")
                print("pip install " + " ".join(missing_packages))
                return False
        else:
            print("Dependencies must be installed to continue. Please install them manually.")
            print("pip install " + " ".join(missing_packages))
            return False
    else:
        print("All dependencies are installed.")

    # Set up environment variables
    print("\nSetting up environment variables...")

    # Check if .env file already exists
    if os.path.exists(".env"):
        print("A .env file already exists.")
        override = input("Do you want to override it? (y/n): ").lower() == 'y'

        if not override:
            print("Keeping existing .env file.")
            return True

    # Get OpenAI API key
    api_key = input("Enter your OpenAI API key: ").strip()

    if not api_key:
        print("No API key provided. You can add it later by editing the .env file.")

    # Create .env file
    success = create_dotenv_file(api_key)

    if success:
        print("\nSetup completed successfully!")
        print("You can now use the Shelter Match RAG system.")
        return True
    else:
        print("\nThere was an error creating the .env file.")
        print("Please create it manually with the following content:")
        print("OPENAI_API_KEY=your_api_key_here")
        return False


if __name__ == "__main__":
    setup_environment()