import sys
from pathlib import Path
import logging
from dotenv import load_dotenv

# Add parent directory to path to allow importing from sibling packages
sys.path.append(str(Path(__file__).parent.parent))

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import our modules
from matching.questionnaire_analyzer import QuestionnaireAnalyzer
from embedding.embedding_pipeline import EmbeddingPipeline


def test_questionnaire_analysis():
    """Test the questionnaire analysis and matching pipeline."""
    print("\n=== Shelter Match RAG - Questionnaire Analysis Test ===\n")

    # Sample questionnaire
    sample_questionnaire = {
        "name": "Jane Smith",
        "email": "jane.smith@example.com",
        "phone": "555-123-4567",
        "address": "123 Main St, Anytown, USA",
        "housing_type": "House with fenced yard",
        "own_or_rent": "Own",
        "landlord_permission": "N/A (Own)",
        "adults_in_home": "2",
        "children_in_home": "2 (ages 8 and 10)",
        "other_pets": "1 cat (5 years old)",
        "work_schedule": "Work from home 3 days/week, office 2 days/week",
        "active_lifestyle": "Moderately active, daily walks, weekend hikes",
        "dog_experience": "Had family dogs growing up, currently don't own a dog",
        "time_alone": "Dog would be alone 4-6 hours on office days",
        "size_preference": "Medium to large",
        "age_preference": "Young adult to adult (2-5 years)",
        "energy_level": "Medium to high energy, playful but trainable",
        "grooming_willingness": "Willing to do regular brushing, occasional professional grooming",
        "training_plans": "Basic obedience classes, consistent home training",
        "special_considerations": "Need a dog that's good with cats and children",
        "adoption_motivation": "Looking for a family companion, exercise partner"
    }

    # Initialize analyzer
    analyzer = QuestionnaireAnalyzer()

    # Analyze questionnaire
    print("Analyzing questionnaire...")
    analysis_result = analyzer.analyze_questionnaire(sample_questionnaire)

    # Print analysis
    print("\n=== Analysis Results ===")
    for key, value in analysis_result["analysis"].items():
        print(f"{key.replace('_', ' ').title()}: {value}")

    # Print search query
    print("\nGenerated Search Query:")
    print(f'"{analysis_result["search_query"]}"')

    print("\nExplanation:")
    print(analysis_result["explanation"])

    # Generate variations
    print("\n=== Query Variations ===")
    variations = analyzer.generate_expanded_queries(analysis_result["search_query"])
    for i, variation in enumerate(variations, 1):
        print(f"{i}. \"{variation}\"")

    # Try to find matches if vector store exists
    try:
        pipeline = EmbeddingPipeline()
        if pipeline.load_latest_vector_store():
            print("\n=== Finding Matches ===")

            # Search with the base query
            print(f"Searching with base query: \"{analysis_result['search_query']}\"")
            base_results = pipeline.search_similar_dogs(analysis_result["search_query"], top_k=3)

            # Display base results
            print("\nTop matches with base query:")
            for i, result in enumerate(base_results, 1):
                score_percentage = int(result['similarity_score'] * 100)
                print(f"{i}. {result['name']} ({result['breed']}) - Match: {score_percentage}%")
                print(f"   Sex: {result['sex']}, Age: {result['age']}, Weight: {result['weight']}")
                print(f"   Description: {result.get('description_snippet', 'N/A')}")
                print("")

            # Try variations if we have any
            if variations:
                best_variation = variations[0]
                print(f"\nSearching with variation: \"{best_variation}\"")
                variation_results = pipeline.search_similar_dogs(best_variation, top_k=3)

                # Display variation results
                print("\nTop matches with variation query:")
                for i, result in enumerate(variation_results, 1):
                    score_percentage = int(result['similarity_score'] * 100)
                    print(f"{i}. {result['name']} ({result['breed']}) - Match: {score_percentage}%")
                    print(f"   Sex: {result['sex']}, Age: {result['age']}, Weight: {result['weight']}")
                    print(f"   Description: {result.get('description_snippet', 'N/A')}")
                    print("")
        else:
            print("\n⚠️ No vector store found. Run the embedding pipeline first to create embeddings.")
    except Exception as e:
        print(f"\n❌ Error finding matches: {e}")
        print("Make sure you've run the embedding pipeline first to create the vector store.")


if __name__ == "__main__":
    test_questionnaire_analysis()