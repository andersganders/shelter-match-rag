import os
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataProcessor:
    """Process and standardize dog data from multiple sources."""

    def __init__(self):
        pass

    def standardize_petpoint_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize data from PetPoint to a common format.

        Args:
            df: DataFrame with PetPoint data

        Returns:
            Standardized DataFrame
        """
        if df.empty:
            return pd.DataFrame()

        # Create a new standardized dataframe
        std_df = pd.DataFrame()

        # Map PetPoint fields to standard fields
        # This mapping would need to be adjusted based on actual PetPoint data structure
        field_mapping = {
            'animalID': 'dog_id',
            'animalName': 'name',
            'animalBreed': 'breed',
            'animalSex': 'sex',
            'animalAge': 'age',
            'animalWeight': 'weight',
            'animalDescription': 'description'
        }

        # Copy and rename fields
        for source_field, target_field in field_mapping.items():
            if source_field in df.columns:
                std_df[target_field] = df[source_field]

        # Add source indicator
        std_df['data_source'] = 'petpoint'
        std_df['source_id'] = df.get('animalID', '')
        std_df['processed_date'] = datetime.now().isoformat()

        return std_df

    def standardize_rescuegroups_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize data from RescueGroups to a common format.

        Args:
            df: DataFrame with RescueGroups data

        Returns:
            Standardized DataFrame
        """
        if df.empty:
            return pd.DataFrame()

        # Create a new standardized dataframe
        std_df = pd.DataFrame()

        # Map RescueGroups fields to standard fields
        # This mapping would need to be adjusted based on actual RescueGroups data structure
        field_mapping = {
            'id': 'dog_id',
            'name': 'name',
            'breedPrimary': 'breed',
            'sex': 'sex',
            'ageGroup': 'age',
            'weightPounds': 'weight',
            'descriptionText': 'description'
        }

        # Copy and rename fields
        for source_field, target_field in field_mapping.items():
            if source_field in df.columns:
                std_df[target_field] = df[source_field]

        # Add source indicator
        std_df['data_source'] = 'rescuegroups'
        std_df['source_id'] = df.get('id', '')
        std_df['processed_date'] = datetime.now().isoformat()

        return std_df

    def extract_dogs_from_message_boards(self, posts_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract dog information from message board posts.
        This is a simple extraction and would need to be enhanced with NLP.

        Args:
            posts_df: DataFrame with message board posts

        Returns:
            DataFrame with extracted dog information
        """
        if posts_df.empty:
            return pd.DataFrame()

        # This is a placeholder for a more sophisticated NLP extraction
        # In a real implementation, you might use Named Entity Recognition
        # and other NLP techniques to extract structured data

        # For now, we'll use a simple approach to extract key information
        extracted_data = []

        for _, post in posts_df.iterrows():
            content = post.get('content', '')
            title = post.get('title', '')

            # Simple extraction of dog names (look for quotes or patterns like "Name is a...")
            name_match = re.search(r'"([^"]+)"|\*([^*]+)\*|(\w+) is a', content)
            name = name_match.group(1) if name_match else "Unknown"

            # Simple breed extraction (look for common breed mentions)
            breed_patterns = [
                r'(Lab(?:rador)?(?:\s+Retriever)?)',
                r'(German Shepherd)',
                r'(Pit Bull)',
                r'(Terrier)',
                r'(Beagle)',
                r'(Chihuahua)',
                r'(Boxer)',
                r'(Poodle)',
                r'(Husky)',
                r'(Golden Retriever)',
                # Add more breeds as needed
            ]

            breed = "Mixed"
            for pattern in breed_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    breed = re.search(pattern, content, re.IGNORECASE).group(1)
                    break

            # Extract age mentions
            age_match = re.search(r'(\d+)(?:\s+)(?:year|month)s?(?:\s+)old', content, re.IGNORECASE)
            age = age_match.group(0) if age_match else "Unknown"

            # Extract sex mentions
            sex = "Unknown"
            if re.search(r'\b(male|boy|he)\b', content, re.IGNORECASE):
                sex = "Male"
            elif re.search(r'\b(female|girl|she)\b', content, re.IGNORECASE):
                sex = "Female"

            extracted_data.append({
                'name': name,
                'breed': breed,
                'age': age,
                'sex': sex,
                'description': content[:200] + "...",  # Truncated description
                'data_source': 'message_board',
                'source_id': str(post.get('source_url', '')) + "/" + str(post.get('title', '')),
                'processed_date': datetime.now().isoformat()
            })

        return pd.DataFrame(extracted_data)

    def merge_and_process_data(self,
                               petpoint_path: str = "data/raw/petpoint_dogs.csv",
                               rescuegroups_path: str = "data/raw/rescuegroups_dogs.csv",
                               message_boards_path: str = "data/raw/message_board_posts.csv",
                               output_path: str = "data/processed/all_dogs.csv") -> str:
        """
        Merge and process data from all sources into a standardized format.

        Args:
            petpoint_path: Path to PetPoint CSV
            rescuegroups_path: Path to RescueGroups CSV
            message_boards_path: Path to message boards CSV
            output_path: Path to save the merged CSV

        Returns:
            Path to the processed file
        """
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Initialize an empty list to store standardized dataframes
        std_dfs = []

        # Process PetPoint data if available
        if os.path.exists(petpoint_path):
            try:
                petpoint_df = pd.read_csv(petpoint_path)
                std_petpoint_df = self.standardize_petpoint_data(petpoint_df)
                std_dfs.append(std_petpoint_df)
                logger.info(f"Processed {len(std_petpoint_df)} dogs from PetPoint")
            except Exception as e:
                logger.error(f"Error processing PetPoint data: {e}")

        # Process RescueGroups data if available
        if os.path.exists(rescuegroups_path):
            try:
                rescuegroups_df = pd.read_csv(rescuegroups_path)
                std_rescuegroups_df = self.standardize_rescuegroups_data(rescuegroups_df)
                std_dfs.append(std_rescuegroups_df)
                logger.info(f"Processed {len(std_rescuegroups_df)} dogs from RescueGroups")
            except Exception as e:
                logger.error(f"Error processing RescueGroups data: {e}")

        # Process message board data if available
        if os.path.exists(message_boards_path):
            try:
                message_boards_df = pd.read_csv(message_boards_path)
                extracted_dogs_df = self.extract_dogs_from_message_boards(message_boards_df)
                std_dfs.append(extracted_dogs_df)
                logger.info(f"Extracted {len(extracted_dogs_df)} dogs from message boards")
            except Exception as e:
                logger.error(f"Error processing message board data: {e}")

        # Concatenate all standardized dataframes
        if std_dfs:
            all_dogs_df = pd.concat(std_dfs, ignore_index=True)

            # Save to CSV
            all_dogs_df.to_csv(output_path, index=False)
            logger.info(f"Saved {len(all_dogs_df)} processed dog records to {output_path}")

            return output_path
        else:
            logger.warning("No data to process from any source")
            return ""