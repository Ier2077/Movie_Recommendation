from db_utils import load_movie_data, index_movie_vectors, save_manifest, batch_embed_texts
from app_utils import init_chromadb_collection

MOVIES_CSV = "movies_metadata.csv"
CHROMA_DB_PATH = "./chroma_db"


def count_indexed_movies(collection):
    """Returns the number of movies/documents indexed in the collection."""
    return collection.count()


def main():
    movies = load_movie_data(MOVIES_CSV)
    client, collection = init_chromadb_collection(CHROMA_DB_PATH)
    movie_vectors = batch_embed_texts(movies)
    index_movie_vectors(collection, movies, movie_vectors)
    save_manifest(movies)
    print(f"Indexed {count_indexed_movies(collection)} movies in ChromaDB.")
    print("✅ Batch multi-vector indexing complete! ChromaDB ready for hybrid search.")
    

if __name__ == "__main__":
    main()
    


