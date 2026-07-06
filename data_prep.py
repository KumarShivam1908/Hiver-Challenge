"""
data_prep.py — Ingestion and Vector DB Setup

Design Decision:
    We use ChromaDB locally to avoid dependency on an external DB service for this
    demo/challenge. The Enron dataset provides real-world corporate context. 
    By chunking at the email level and embedding the 'EmailSend' (incoming email),
    we can find historically similar emails when a new one arrives.
"""

import os
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
from chromadb.config import Settings
import config

def initialize_database():
    print(f"Loading data from: {config.DATA_CSV}")
    if not os.path.exists(config.DATA_CSV):
        raise FileNotFoundError(f"Dataset not found at {config.DATA_CSV}. Please ensure it exists.")

    try:
        df = pd.read_csv(config.DATA_CSV)
    except Exception as e:
        print(f"Failed to load CSV: {e}")
        return

    # 1. Clean Data: Drop rows where send or reply is missing
    df = df.dropna(subset=['EmailSend', 'EmailReply'])
    
    # 2. Sample Data for speed
    if len(df) > config.DB_SAMPLE_SIZE:
        df = df.sample(n=config.DB_SAMPLE_SIZE, random_state=42).reset_index(drop=True)
        
    print(f"Ingesting {len(df)} emails into vector DB...")

    # 3. Initialize ChromaDB
    os.makedirs(config.CHROMA_DIR, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=config.CHROMA_DIR)
    
    # Reset collection if running this multiple times
    collection_name = "enron_emails"
    try:
        chroma_client.delete_collection(name=collection_name)
    except Exception:
        pass # Collection might not exist yet
        
    collection = chroma_client.create_collection(name=collection_name)
    
    # 4. Load Embedding Model
    print(f"Loading embedding model: {config.EMBED_MODEL}")
    model = SentenceTransformer(config.EMBED_MODEL)
    
    # 5. Batch Process and Embed
    # Why batch process?
    #   • Prevents memory exhaustion on larger datasets
    #   • Provides progress updates to the user
    batch_size = 100
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        
        # What do we embed? The incoming email text
        documents = batch['EmailSend'].tolist()
        
        # What metadata do we store? The reply, subject, and date
        metadatas = [
            {
                "email_reply": str(row['EmailReply']),
                "subject_send": str(row.get('SubjectSend', '')),
                "date_send": str(row.get('DateSend', ''))
            } for _, row in batch.iterrows()
        ]
        
        # Unique IDs
        ids = [f"email_{idx}" for idx in range(i, i + len(batch))]
        
        # Generate Embeddings
        embeddings = model.encode(documents).tolist()
        
        collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        print(f"-> Processed {i + len(batch)} / {len(df)} items")
        
    print("Database successfully initialized!")

if __name__ == "__main__":
    initialize_database()
