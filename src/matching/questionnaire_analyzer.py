import os
import logging
import json
from typing import Dict, List, Any, Optional
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class QuestionnaireAnalyzer:
    """Analyzes adopter questionnaires and converts them to search queries."""

    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        """
        Initialize the questionnaire analyzer.

        Args:
            model_name: Name of the OpenAI model to use
        """
        self.model_name = model_name

        # Set up OpenAI client
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            logger.warning("OpenAI API key not found. Questionnaire analysis will not work without it.")
            self.client = None
        else:
            self.client = OpenAI(api_key=openai_api_key)

    def analyze_questionnaire(self, questionnaire: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a completed questionnaire and generate a search query.

        Args:
            questionnaire: Dictionary with questionnaire responses

        Returns:
            Dictionary with analysis results and search query
        """
        if not self.client:
            logger.error("OpenAI client not initialized. Check API key.")
            return {
                "analysis": {
                    "error": "OpenAI API key not found or invalid"
                },
                "search_query": "Looking for a dog that matches the adopter's preferences",
                "explanation": "Analysis could not be completed due to missing API key"
            }

        # Convert questionnaire to a string format
        questionnaire_text = self._format_questionnaire(questionnaire)

        # Define the system prompt
        system_prompt = """
        You are an AI assistant helping match potential adopters with shelter dogs.
        Your task is to analyze a potential adopter's questionnaire and extract key information about:

        1. Their living situation (apartment, house, yard, etc.)
        2. Their lifestyle (active, sedentary, work hours, etc.)
        3. Their experience with dogs
        4. Their preferences for dog characteristics (size, age, energy level, etc.)
        5. Any special requirements or constraints

        Then, create a search query that would help find the most suitable dogs for this adopter.

        Provide your response in the following JSON format:
        {
            "analysis": {
                "living_situation": "Brief description of their living situation",
                "lifestyle": "Brief description of their lifestyle",
                "experience": "Brief description of their experience with dogs",
                "preferences": "Brief description of their preferences",
                "constraints": "Brief description of any constraints"
            },
            "search_query": "A natural language search query to find suitable dogs",
            "explanation": "Brief explanation of why this search query would find good matches"
        }
        """

        try:
            # Call the OpenAI API
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Here is the completed questionnaire:\n\n{questionnaire_text}"}
                ],
                temperature=0.3,
                max_tokens=1000
            )

            # Extract and parse the response
            result_text = response.choices[0].message.content.strip()

            try:
                # Try to parse as JSON
                result = json.loads(result_text)
                return result
            except json.JSONDecodeError:
                # If not valid JSON, return a basic result
                logger.error("Failed to parse analyzer response as JSON")
                return {
                    "analysis": {
                        "error": "Failed to parse analyzer response"
                    },
                    "search_query": "Looking for a dog that matches the adopter's preferences",
                    "explanation": "Analysis could not be completed properly"
                }

        except Exception as e:
            logger.error(f"Error analyzing questionnaire: {e}")
            return {
                "analysis": {
                    "error": f"Error: {str(e)}"
                },
                "search_query": "Looking for a dog that matches the adopter's preferences",
                "explanation": "Analysis could not be completed due to an error"
            }

    def _format_questionnaire(self, questionnaire: Dict[str, Any]) -> str:
        """
        Format a questionnaire dictionary as a human-readable string.

        Args:
            questionnaire: Dictionary with questionnaire responses

        Returns:
            Formatted string representation of the questionnaire
        """
        formatted = []

        for key, value in questionnaire.items():
            # Convert keys from snake_case or camelCase to readable form
            readable_key = key.replace('_', ' ').title()

            formatted.append(f"{readable_key}: {value}")

        return "\n".join(formatted)

    def generate_expanded_queries(self, base_query: str, num_variations: int = 3) -> List[str]:
        """
        Generate expanded variations of a base search query.

        Args:
            base_query: The base search query
            num_variations: Number of variations to generate

        Returns:
            List of expanded query variations
        """
        if not self.client:
            logger.error("OpenAI client not initialized. Check API key.")
            return [base_query]

        system_prompt = """
        You are an AI assistant helping match potential adopters with shelter dogs.
        Your task is to generate variations of a base search query to improve the chances of finding good matches.

        Create variations that:
        1. Emphasize different aspects of the original query
        2. Use different synonyms or phrasings
        3. Focus on different potential priorities

        Each variation should still accurately reflect the original query's intent.
        """

        try:
            # Call the OpenAI API
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",
                     "content": f"Base query: \"{base_query}\"\n\nGenerate {num_variations} variations of this query."}
                ],
                temperature=0.7,
                max_tokens=500
            )

            # Extract and parse the response
            result_text = response.choices[0].message.content.strip()

            # Split the response into lines and filter out non-queries
            lines = result_text.split('\n')
            queries = []

            for line in lines:
                # Remove numbering, quotes, and other formatting
                clean_line = line.strip()

                # Remove numbering (e.g., "1." or "Variation 1:")
                for prefix in [f"{i}. " for i in range(1, 10)] + [f"Variation {i}: " for i in range(1, 10)]:
                    if clean_line.startswith(prefix):
                        clean_line = clean_line[len(prefix):]
                        break

                # Remove quotes
                clean_line = clean_line.strip('"\'')

                if clean_line and clean_line != base_query and len(clean_line) > 10:
                    queries.append(clean_line)

            # Add the original query if we have space
            if not queries or (len(queries) < num_variations):
                queries.insert(0, base_query)

            # Return the requested number of variations (or all if fewer)
            return queries[:num_variations]

        except Exception as e:
            logger.error(f"Error generating query variations: {e}")
            return [base_query]  # Return the original query on error