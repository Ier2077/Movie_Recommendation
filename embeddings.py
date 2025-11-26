from openai import AzureOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version= "2024-05-01-preview"
)

DEPLOYMENT = os.getenv("AZURE_EMBEDDING_DEPLOYMENT")

def create_embedding(text):
    result = client.embeddings.create(
        model=DEPLOYMENT,
        input=text
    )
    return result.data[0].embedding


