import os
import chromadb
import numpy as np
from openai import AzureOpenAI
from dotenv import load_dotenv
from db_utils import movie_finder
import json
from history_manager import load_conversation_history, save_conversation_history, clear_conversation_history
from app_utils import init_azure_client, init_chromadb_collection
from Function_calling import run_llm_with_function_call



# --- 1. SETUP ---
print("--- Initializing Recommendation App ---")

# Load all environment variables
from dotenv import load_dotenv
load_dotenv()

azure_client, AZURE_DEPLOYMENT_NAME = init_azure_client()

# Initialize ChromaDB Client to access our "Content Catalog"
collection = init_chromadb_collection()

def run_app_file(user_input):
    
    run_llm_with_function_call(user_input)

if __name__ == "__main__":
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        run_app_file(user_input)