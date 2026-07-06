"""
generator.py — Core RAG Pipeline

Design Decision:
    We use Few-Shot Prompting over past emails rather than fine-tuning an LLM.
    Why?
      1. Cost & Speed: No training infrastructure required.
      2. Dynamic: We can swap the dataset (e.g., from Enron to customer support) 
         without retraining.
      3. Explainability: We know exactly which examples the LLM is using.
"""

import sys
import chromadb
from groq import Groq
from sentence_transformers import SentenceTransformer
import config

class EmailGenerator:
    def __init__(self):
        """Initializes the external clients and models."""
        self.groq_client = self._init_groq()
        self.chroma_collection, self.embedder = self._init_retrieval()

    def _init_groq(self):
        if not config.GROQ_API_KEY:
            print("Warning: GROQ_API_KEY not found in config/env.", file=sys.stderr)
            return None
        try:
            return Groq(api_key=config.GROQ_API_KEY)
        except Exception as e:
            print(f"Failed to initialize Groq: {e}", file=sys.stderr)
            return None

    def _init_retrieval(self):
        try:
            client = chromadb.PersistentClient(path=config.CHROMA_DIR)
            # We assume collection exists (created by data_prep.py)
            collection = client.get_collection("enron_emails")
            embedder = SentenceTransformer(config.EMBED_MODEL)
            return collection, embedder
        except Exception as e:
            print(f"Warning: Failed to initialize Vector DB. Did you run data_prep.py? Error: {e}", file=sys.stderr)
            return None, None

    def retrieve_context(self, incoming_email: str) -> list:
        """Retrieves K most similar emails from ChromaDB."""
        if not self.chroma_collection or not self.embedder:
            return []
            
        # Embed the query
        query_embedding = self.embedder.encode([incoming_email]).tolist()
        
        # Search DB
        results = self.chroma_collection.query(
            query_embeddings=query_embedding,
            n_results=config.TOP_K
        )
        
        examples = []
        if results['documents'] and len(results['documents'][0]) > 0:
            docs = results['documents'][0]
            metadatas = results['metadatas'][0]
            
            for i in range(len(docs)):
                examples.append({
                    "incoming": docs[i],
                    "reply": metadatas[i].get("email_reply", "No reply found")
                })
        return examples

    def generate_reply(self, incoming_email: str) -> tuple:
        """Generates a reply using retrieved context (RAG)."""
        if not self.groq_client:
            return "Error: Groq client not initialized. Check API Key.", []

        # 1. Retrieve
        examples = self.retrieve_context(incoming_email)
        
        # 2. Augment (Construct Prompt)
        system_prompt = (
            "You are an expert executive assistant drafting email replies.\n"
            "Below are examples of how this user typically responds to emails. "
            "Mimic their tone, brevity, and format. Do NOT include placeholders like [Your Name].\n"
        )
        
        if examples:
            system_prompt += "\n--- HISTORICAL EXAMPLES ---\n"
            for i, ex in enumerate(examples):
                system_prompt += f"\nExample {i+1}:\n"
                system_prompt += f"Incoming: {ex['incoming']}\n"
                system_prompt += f"Reply: {ex['reply']}\n"
            system_prompt += "\n--- END OF EXAMPLES ---\n"

        system_prompt += "\nWrite a direct, professional reply to the new incoming email based on the context."

        user_prompt = f"New Incoming Email:\n{incoming_email}"
        
        # 3. Generate
        try:
            response = self.groq_client.chat.completions.create(
                model=config.GROQ_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7, # A bit of creativity, but mostly deterministic
                max_tokens=400
            )
            return response.choices[0].message.content.strip(), examples
        except Exception as e:
            return f"Generation Error: {e}", examples

# Singleton instance for easy import
_generator = None
def get_generator():
    global _generator
    if _generator is None:
        _generator = EmailGenerator()
    return _generator
