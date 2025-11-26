from chromadb import PersistentClient

# point to your existing Chroma folder
client = PersistentClient(path="./chroma_db")

# list collections
collections = client.list_collections()
print("Collections:", collections)

for col in collections:
    c = client.get_collection(col.name)
    print(f"{col.name} has {c.count()} embeddings")
