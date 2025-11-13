import os
import chromadb
import numpy as np
from openai import AzureOpenAI
from dotenv import load_dotenv
from db_utils import movie_finder



# --- 1. SETUP ---
print("--- Initializing Recommendation App ---")

# Load all environment variables
load_dotenv()

# Initialize Azure OpenAI Client for the "LLM Generator"
try:
    azure_client = AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview")
    )
    
    AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    if not AZURE_DEPLOYMENT_NAME:
        raise EnvironmentError("AZURE_OPENAI_DEPLOYMENT_NAME not set in .env")
except Exception as e:
    print(f"Error initializing Azure OpenAI client: {e}")
    exit()

# Initialize ChromaDB Client to access our "Content Catalog"
try:
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection(name="movies")
    print(f"Successfully connected to Vector DB. Total movies indexed: {collection.count()}")
except Exception as e:
    print(f"Error connecting to ChromaDB: {e}")
    exit()


def run_app_file(user_input):
    """Main function that accepts user input and runs the recommendation engine."""

# --- 2. USER INPUT ---
# This is our mock user. They have "liked" these movies.
# We use the imdb_id's for the movies we like
    try :
        USER_LIKES = [title.strip() for title in user_input.split(",") if title.strip()]
        if not USER_LIKES:
             USER_LIKES = ["Heat", "GoldenEye", "Sudden Death"]
    
    except Exception as e:
        print(f"Error reading user likes: {e}")
        exit()
    print(f"\nMock User Likes: {', '.join(USER_LIKES)}")


# --- 3. USER PROFILE EMBEDDING ---
    print("Creating User Profile Embedding (User Taste Vector)...")
    USER_LIKES_ids = movie_finder(USER_LIKES, collection)

    if not USER_LIKES_ids:
        print("Error: Could not find any of the user's liked movies in the database.")
        return

    # Get the vectors for the movies the user likes
    try:
        liked_movies = collection.get(ids=USER_LIKES_ids, include=["embeddings"])
        liked_vectors = liked_movies.get('embeddings')

        # Convert to a NumPy array and ensure we actually have embeddings before averaging
        liked_vectors_array = np.array(liked_vectors)
        if liked_vectors_array.size == 0:
            print("Error: Could not find vectors for liked movies. Did build_index.py run correctly?")
            exit()

        # Create the "User Taste Vector" by averaging the liked vectors
        user_taste_vector = np.mean(liked_vectors_array, axis=0).tolist()
        print("Successfully created user taste vector.")
    except Exception as e:
        print(f"Error creating user profile: {e}")
        exit()


# --- 4. RETRIEVER ---
    print("Running Retriever to find Top-k Candidates...")

    # Find the 5 movies closest to the user's taste vector
    # We add 3 to n_results because the 3 movies they *like* will be the closest
    k = 5
    results = collection.query(
        query_embeddings=[user_taste_vector],
        n_results=k + len(USER_LIKES_ids) # Ask for more to filter out seen movies
    )

    # Filter out movies the user has already seen (USER_LIKES)
    candidates = []
    for i, movie_id in enumerate(results['ids'][0]):
        if movie_id not in USER_LIKES_ids:
            candidate_info = {
                "title": results['metadatas'][0][i]['title'],
                "genre": results['metadatas'][0][i]['genre'],
                "plot": results['documents'][0][i]
            }
            candidates.append(candidate_info)
        
        # Stop once we have k candidates
        if len(candidates) >= k:
            break

    print(f"Found {len(candidates)} new candidates.")


# --- 5. LLM GENERATOR ---
    print("Sending candidates to LLM Generator (Azure OpenAI)...")

    # Build the prompt for the LLM
    system_prompt = """
You are a helpful and enthusiastic movie recommendation assistant. 
Your job is to rank a list of candidate movies based on a user's known preferences.
You must explain *why* each movie is a good recommendation, connecting it to the user's liked movies.
Present the output as a ranked list (1, 2, 3...).
"""

    # Format the user's likes and the candidates for the prompt
    liked_movies_str = ", ".join(USER_LIKES)
    candidate_movies_str = "\n".join([
        f"- Title: {c['title']} (Genre: {c['genre']})\n  Plot: {c['plot']}" for c in candidates
    ])

    user_prompt = f"""
Here's what I know about the user:
- They LIKE: {liked_movies_str}

Here are the top candidates you found:
{candidate_movies_str}

Please rank these candidates from 1 to {len(candidates)} and explain why you are recommending each one based on the user's tastes.
"""

    # Send to Azure OpenAI
    try:
        response = azure_client.chat.completions.create(
            model=AZURE_DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7
        )
        
        final_recommendations = response.choices[0].message.content

        # --- 6. OUTPUT ---
        print("\n" + "="*50)
        print(" Personalized Recommendations")
        print("="*50 + "\n")
        print(final_recommendations)

    except Exception as e:
        print(f"\nAn error occurred with the Azure OpenAI request: {e}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        user_input = " ".join(sys.argv[1:])
        run_app_file(user_input)
    else:
        user_input = input("Enter a list of movie titles you like (comma-separated): ")
        run_app_file(user_input)