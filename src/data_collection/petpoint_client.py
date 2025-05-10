import os
from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd
from .base_client import BaseClient, logger


class PetPointClient(BaseClient):
    """Client for interacting with the PetPoint API."""

    def __init__(self):
        api_url = os.getenv("PETPOINT_API_URL")
        api_key = os.getenv("PETPOINT_API_KEY")
        if not api_url:
            logger.warning("PETPOINT_API_URL environment variable not set")
            api_url = "https://api.petpoint.com/api/v1"  # Default URL, may need to be updated

        super().__init__(api_url, api_key)

    def get_available_dogs(self, status: str = "available", limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get a list of available dogs from PetPoint.

        Args:
            status: Filter by animal status (available, adopted, etc.)
            limit: Maximum number of records to return

        Returns:
            List of dog records
        """
        params = {
            "species": "dog",
            "status": status,
            "limit": limit
        }

        response = self.get("animals", params)
        if "error" in response:
            logger.error(f"Failed to get dogs: {response['error']}")
            return []

        return response.get("animals", [])

    def get_dog_details(self, animal_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific dog.

        Args:
            animal_id: The PetPoint ID of the animal

        Returns:
            Detailed dog information or None if not found
        """
        response = self.get(f"animals/{animal_id}")
        if "error" in response:
            logger.error(f"Failed to get dog details for ID {animal_id}: {response['error']}")
            return None

        return response

    def fetch_and_save_dogs(self, output_path: str = "data/raw/petpoint_dogs.csv") -> str:
        """
        Fetch all available dogs and save to CSV.

        Args:
            output_path: Path to save the CSV file

        Returns:
            Path to the saved file
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        dogs = self.get_available_dogs(limit=500)  # Adjust limit as needed

        if not dogs:
            logger.warning("No dogs retrieved from PetPoint")
            return ""

        # Convert to DataFrame
        df = pd.DataFrame(dogs)

        # Add timestamp
        df['fetch_date'] = datetime.now().isoformat()
        df['source'] = 'petpoint'

        # Save to CSV
        df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(df)} dogs to {output_path}")

        return output_path