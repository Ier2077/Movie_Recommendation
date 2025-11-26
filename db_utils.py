import chromadb
import numpy as np
import pandas as pd
import os   
import time
from embeddings import create_embedding
import json


#configure and connect to ChromaDB
movies_csv = "movies_metadata.csv"
chromadb_path = "./chroma_db"
top_k = 5000
batch_size = 100
MANIFEST_FILE = "chroma_manifest.json"


# Load and clean movie data
def load_movie_data(csv_file, k = top_k):
    df = pd.read_csv(csv_file, low_memory=False)
    df = df.dropna(subset=['overview', 'imdb_id', 'title','popularity'])
    df = df.drop_duplicates(subset=['imdb_id'])
    df = df[df['imdb_id'].str.startswith('tt')]
    df['popularity'] = pd.to_numeric(df['popularity'], errors='coerce')
    df = df.dropna(subset=['popularity'])
    df = df.sort_values(by='popularity', ascending=False).head(k)
    return df.to_dict(orient='records')
    

# Initialize ChromaDB Client   
def init_chromadb_client(db_path=chromadb_path):
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(name="movies")
    return client, collection
    
# create dictionary of metadatas for embedding
def text_for_embedding(movie):
    """ Returns a dict of texts to embed for each movie.
    Each key corresponds to a vector type."""
    texts ={}
    #plot
    plot = movie.get('overview', '')
    texts['plot'] = f"Plot: {plot}"
    
    #cast,directer
    cast = movie.get('cast', '')
    director = movie.get('director', '')
    texts['cast_director'] = f"Cast: {cast}. \nDirector: {director}. "
    
    #title, genre, popularity 
    title = movie['title']
    genre = str(movie.get('genres', ''))
    popularity = movie.get('popularity', '')
    texts['title_genre_popularity'] = f"Title: {title}. \nGenre: {genre}. \nPopularity: {popularity}. Plot: {movie['overview']}"
    
    return texts
    

    
# Batch indexing
def batch_embed_texts(movies, batch_size= batch_size):
    """
    Returns a list of dicts per movie:
    [{"plot": vec1, "cast_director": vec2, "title_genre": vec3}, ...]
    """
    all_vectors = []
    for i in range(0, len(movies), batch_size):
        batch = movies[i:i+batch_size]
        for movie in batch:
            texts = text_for_embedding(movie)
            vectors = {key: create_embedding(text) for key, text in texts.items()}
            all_vectors.append(vectors)
        print(f"Embedded {min(i+batch_size, len(movies))}/{len(movies)} movies...")
        time.sleep(0.5)  #avoid rate limits
    return all_vectors



#movie indexer function
def index_movie_vectors(collection,movies,movie_vectors,batch_size = batch_size):
    
    """Indexes the given movie vectors into the ChromaDB collection."""
    for i in range(0, len(movies), batch_size):
        batch_movies = movies[i:i+batch_size]
        batch_vectors = movie_vectors[i:i+batch_size]
        
        ids = [movie['imdb_id'] for movie in batch_movies]
        # Concatenate per-field vectors into a single flat embedding for each movie.
        # This is compatible with ChromaDB versions that expect one vector per document.
        embeddings = [
            np.concatenate([
                np.array(vecs['title_genre_popularity']),
                np.array(vecs['plot']),
                np.array(vecs['cast_director'])
            ]).tolist()
            for vecs in batch_vectors
        ]
        
        documents = [movie['overview'] for movie in batch_movies]
        metadatas = [{
        "title": movie['title'],
        "title_lower": movie['title'].lower(),
        "genre": str(movie.get('genres', '')),
        "popularity": movie.get('popularity', ''),
        "vector_parts": "title_genre_popularity,plot,cast_director"  
    } for movie in batch_movies]
        
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        print(f"Indexed {min(i+batch_size, len(movies))}/{len(movies)} movies...")


#function to save manifest file
def save_manifest(movies, version="v2"):
    manifest = {
        "version": version,
        "total_movies": len(movies),
        "fields": ["plot","cast_director","title_genre_popularity"],
        "ids": [m['imdb_id'] for m in movies]
    }
    with open(MANIFEST_FILE, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=4, ensure_ascii=False)
    print(f"Saved manifest to {MANIFEST_FILE}")




#function to find movie IDs given titles
from rapidfuzz import process, fuzz
def movie_finder(titles, collection):
    '''Given a list of movie titles, return their corresponding IDs from the ChromaDB collection.
    '''
    if not titles or not isinstance(titles, list):
        return []
    resp = collection.get(include=["metadatas"])
    db_ids = resp.get("ids", [])
    db_metadatas = resp.get("metadatas", [])
    
    #mapping title_lower to id
    title_map = {}
    for i, meta in enumerate(db_metadatas):
        title_lower = meta.get('title_lower') or meta.get('title', '').lower()
        title_map[title_lower] = db_ids[i]
    
    #fuzzy matching
    found_ids = []
    for title in titles:
        title_lower = title.lower()
        #find best match
        best_match = process.extractOne(
            title_lower,
            title_map.keys(),
            scorer=fuzz.WRatio,
            score_cutoff=80  #threshold for matching
        )
        if best_match:
            matched_title = best_match[0]
            found_ids.append(title_map[matched_title])
            
    return found_ids


def get_movie_details(movie_ids, collection):
    """
    Retrieve full movie details for given IMDb IDs.
    
    Args:
        movie_ids: List of IMDb IDs (e.g., ["tt0133093", "tt0234215"])
        collection: ChromaDB collection
        
    Returns:
        List of dictionaries with movie details (title, overview, genre, popularity)
    """
    if not movie_ids or not isinstance(movie_ids, list):
        return []
    
    try:
        # Get movies by IDs
        results = collection.get(
            ids=movie_ids,
            include=["metadatas", "documents"]
        )
        
        movie_details = []
        for i, movie_id in enumerate(results.get("ids", [])):
            metadata = results["metadatas"][i]
            document = results["documents"][i]
            
            movie_details.append({
                "imdb_id": movie_id,
                "title": metadata.get("title", "Unknown"),
                "overview": document,
                "genre": metadata.get("genre", ""),
                "popularity": metadata.get("popularity", "")
            })
        
        return movie_details
    except Exception as e:
        print(f"Error retrieving movie details: {e}")
        return []


def find_similar_movies(movie_ids, collection, n_results=5):
    """
    Find similar movies using vector similarity search (RAG).
    
    Args:
        movie_ids: List of IMDb IDs to use as seed movies
        collection: ChromaDB collection
        n_results: Number of similar movies to return (default: 5)
        
    Returns:
        List of similar movies with full details, excluding the seed movies
    """
    if not movie_ids or not isinstance(movie_ids, list):
        return []
    
    try:
        # Get embeddings for the seed movies
        seed_movies = collection.get(
            ids=movie_ids,
            include=["embeddings"]
        )
        
        embeddings = seed_movies.get("embeddings")
        if embeddings is None or len(embeddings) == 0:
            print("No embeddings found for the provided movie IDs")
            return []
        
        # Average the embeddings if multiple seed movies
        if len(embeddings) > 1:
            avg_embedding = np.mean(embeddings, axis=0).tolist()
        else:
            avg_embedding = embeddings[0]
        
        # Query for similar movies
        # Request more results to account for filtering out seed movies
        results = collection.query(
            query_embeddings=[avg_embedding],
            n_results=n_results + len(movie_ids),
            include=["metadatas", "documents", "distances"]
        )
        
        similar_movies = []
        for i, movie_id in enumerate(results["ids"][0]):
            # Skip if this is one of the seed movies
            if movie_id in movie_ids:
                continue
            
            metadata = results["metadatas"][0][i]
            document = results["documents"][0][i]
            distance = results["distances"][0][i]
            
            similar_movies.append({
                "imdb_id": movie_id,
                "title": metadata.get("title", "Unknown"),
                "overview": document,
                "genre": metadata.get("genre", ""),
                "popularity": metadata.get("popularity", ""),
                "similarity_score": 1 - distance  # Convert distance to similarity
            })
            
            # Stop once we have enough results
            if len(similar_movies) >= n_results:
                break
        
        return similar_movies
    except Exception as e:
        print(f"Error finding similar movies: {e}")
        return []
