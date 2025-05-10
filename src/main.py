import os
from dotenv import load_dotenv

load_dotenv()

def main():
    print("Shelter Match RAG - Initial Setup Complete")

    #verify key environment variables (will be used later)
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        print("OpenAI API key found")
    else:
        print("Warning: OpenAI API key not found. You'll need this for the RAG system.")

if __name__ == "__main__":
    main()