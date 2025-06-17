import os
from dotenv import load_dotenv
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import multiprocessing

# Load environment variables
load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")
MONGODB_DB = os.getenv("MONGODB_DB")
MONGODB_COLLECTION = os.getenv("MONGODB_COLLECTION")

def get_mongo_collection():
    client = MongoClient(MONGODB_URI)
    db = client[MONGODB_DB]
    return db[MONGODB_COLLECTION]

def process_batch(args):
    batch, model_name, mongo_uri, db_name, collection_name = args
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name, device=device)
    client = MongoClient(mongo_uri)
    collection = client[db_name][collection_name]
    texts = [
        f"{doc.get('Crime Description', '')} {doc.get('City', '')} {doc.get('Crime Domain', '')}"
        for doc in batch
    ]
    embeddings = model.encode(texts, show_progress_bar=False, device=device)
    for doc, emb in zip(batch, embeddings):
        collection.update_one({"_id": doc["_id"]}, {"$set": {"vector": emb.tolist()}})
    return len(batch)

def main():
    collection = get_mongo_collection()
    model_name = "sentence-transformers/all-MiniLM-L6-v2"

    # Download the model once before multiprocessing to avoid repeated downloads
    print("Downloading the model (this will happen only once)...")
    SentenceTransformer(model_name)

    docs = list(collection.find({"vector": {"$exists": False}}))
    print(f"Found {len(docs)} documents to process.")

    batch_size = 64
    num_workers = min(multiprocessing.cpu_count(), 8)
    batches = [docs[i:i+batch_size] for i in range(0, len(docs), batch_size)]
    args_list = [
        (batch, model_name, MONGODB_URI, MONGODB_DB, MONGODB_COLLECTION)
        for batch in batches
    ]

    with multiprocessing.Pool(processes=num_workers) as pool:
        list(tqdm(pool.imap_unordered(process_batch, args_list), total=len(batches)))

    print("Embedding generation and update complete.")

if __name__ == "__main__":
    main()
