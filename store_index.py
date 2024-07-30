from src.helper import load_pdf, text_split, download_hugging_face_embeddings

import pinecone
from pinecone import Pinecone
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')

extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()

texts = [t.page_content for t in text_chunks]
embedded_texts = [embeddings.embed_query(text) for text in texts]

# Prepare the vectors for upsertion
vectors = [
    {
        "id": str(i),
        "values": embedded_texts[i],
        "metadata": {"text": texts[i]}
    }
    for i in range(len(texts))
]
pc = Pinecone(
    api_key=PINECONE_API_KEY  # Directly pass your API key as a string
)
index_name = "medbot"
batch_size = 10
index = pc.Index(index_name)

for i in range(0, len(vectors), batch_size):
    batch = vectors[i:i + batch_size]
    index.upsert(vectors=batch, namespace="ns1")
    print(f"Batch {i // batch_size + 1} inserted successfully.")

print("All documents inserted successfully.")


