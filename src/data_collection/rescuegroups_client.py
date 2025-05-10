import os
from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd
from .base_client import BaseClient, logger


class RescueGroupsClient(BaseClient):
    """Client for interacting with the RescueGroups.org API."""

    def __init__(self):
        api_url = os.getenv("RESCUEGROUPS_API_URL", "https://api.rescuegroups.org/v5")
        api_key = os.getenv("RESCUEGROUPS_API_KEY")

        if not api_key:
            logger.warning("RESCUEGROUPS_API_KEY environment variable not set")

        super().__init__(api_url, api_key)

        # RescueGroups.org uses a different auth header
        if api_key:
            self.session.headers.update({
                "Authorization": api_key,
                "Content-Type": "application/vnd.api+json"
            })

    def get_available_dogs(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get a list of available dogs from RescueGroups.org.

        Args:
            limit: Maximum number of records to return

        Returns:
            List of dog records
        """
        data = {
            "data": {
                "filters": [
                    {
                        "fieldName": "statuses.name",
                        "operation": "equals",
                        "criteria": "Available"
                    },
                    {
                        "fieldName": "species.singular",
                        "operation": "equals",
                        "criteria": "Dog"
                    }
                ],
                "limit": limit
            }
        }

        response = self.post("public/animals/search", data)
        if "error" in response:
            logger.error(f"Failed to get dogs: {response['error']}")
            return []

        return response.get("data", [])

    def get_dog_details(self, animal_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific dog.

        Args:
            animal_id: The RescueGroups ID of the animal

        Returns:
            Detailed dog information or None if not found
        """
        response = self.get(f"public/animals/{animal_id}")
        if "error" in response:
            logger.error(f"Failed to get dog details for ID {animal_id}: {response['error']}")
            return None

        return response.get("data", {})

    def fetch_and_save_dogs(self, output_path: str = "data/raw/rescuegroups_dogs.csv") -> str:
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
            logger.warning("No dogs retrieved from RescueGroups")
            return ""

        # RescueGroups returns data in a nested format, flatten it
        flattened_dogs = []
        for dog in dogs:
            flat_dog = {
                "id": dog.get("id"),
                "source_id": "rescuegroups"
            }

            # Extract attributes
            attributes = dog.get("attributes", {})
            for key, value in attributes.items():
                flat_dog[key] = value

            flattened_dogs.append(flat_dog)

        # Convert to DataFrame
        df = pd.DataFrame(flattened_dogs)

        # Add timestamp
        df['fetch_date'] = datetime.now().isoformat()
        df['source'] = 'rescuegroups'

        # Save to CSV
        df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(df)} dogs to {output_path}")

        return output_path