import time
import chromadb
import pandas as pd
from embeddings import create_embedding

# --- 1. INITIALIZE VECTOR DB ---
print("Initializing ChromaDB client...")
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="movies")
print("Initialization complete.")

# --- 2. LOAD & CLEAN KAGGLE DATA ---
print("Loading movies_metadata.csv...")
try:
    df = pd.read_csv("movies_metadata.csv", low_memory=False)
except FileNotFoundError:
    print("Error: movies_metadata.csv not found.")
    print("Please download it from Kaggle and place it in the project folder.")
    exit()

# Cleaning the data
# We need 'imdb_id', 'title', and 'overview' (which is the plot)
df = df.dropna(subset=['overview', 'imdb_id', 'title'])
df = df.drop_duplicates(subset=['imdb_id'])
df = df[df['imdb_id'].str.startswith('tt')] # Keep only valid IMDB IDs

# --- !!! SAFETY LIMIT !!! ---
# Process only the first 100 movies to test.
# Remove this line to process all 45,000+ movies
df = df.head(100)
_100_movies = df.head(100)

print(f"Loaded and cleaned data. Processing {len(df)} movies...")

# --- 3. FETCH, EMBED, AND STORE (The Indexing Pipeline) ---
print("Starting content catalog indexing pipeline...")

for index, row in df.iterrows():
    imdb_id = row['imdb_id']
    title = row['title']
    plot = row['overview']
    genre = str(row.get('genres', '')) # Get genre if it exists

    # Check if we already indexed this movie
    if collection.get(ids=[imdb_id])['ids']:
        print(f"Skipping {imdb_id} ({title}): Already in database.")
        continue
    
    # Create the text for embedding
    text_to_embed = f"Title: {title}. Genre: {genre}. Plot: {plot}"

    print(f"Embedding: {imdb_id} ({title})")
    try:
        vector = create_embedding(text_to_embed)

        collection.add(
            embeddings=[vector],
            documents=[plot],
            metadatas=[{
                "title": title,
                "genre": genre
            }],
            ids=[imdb_id] # Use IMDB ID as the unique key
        )
        print(f"Successfully added {imdb_id} to the database.")

    except Exception as e:
        print(f"!!! An error occurred for {imdb_id}: {e} !!!")

    # delay to Azure API to avoid rate limits
    print("Pausing for 1 second...")
    time.sleep(1) # 1-second pause

print("-" * 30)
print("Content Indexing Pipeline COMPLETE.")
print(f"Total movies in database: {collection.count()}")
print("-" * 30)

# Save the selected columns to a .txt file
_100_movies[['imdb_id', 'title']].to_csv("output_movies.txt", sep='\t', index=False, header=True)
print("Saved imdb_id and title to output_movies.txt")
        