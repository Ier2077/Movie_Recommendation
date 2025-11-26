import os
from dotenv import load_dotenv
from openai import AzureOpenAI
import chromadb
load_dotenv()



def init_azure_client():
    try:
        client = AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview")
        )
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        if not deployment:
            raise EnvironmentError("AZURE_OPENAI_DEPLOYMENT_NAME not set in .env")
        return client, deployment
    except Exception as e:
        print(f"Error initializing Azure OpenAI client: {e}")
        exit()
        
def init_chromadb_collection(db_path="./chroma_db"):
    """Return (client, collection) — supports multi-vector collection."""
    try:
        client = chromadb.PersistentClient(path=db_path)
        # Create (or get) a standard collection. Older/newer chromadb
        # installations may not support `vector_names`, so keep a single-vector
        # collection and store concatenated embeddings.
        collection = client.get_or_create_collection(
            name="movies",
            metadata={"description": "movie collection (concatenated embeddings)"}
        )

        print(f"Connected to ChromaDB: {collection.count()} entries")
        return client, collection
    except Exception as e:
        print(f"Error connecting to ChromaDB: {e}")
        raise
