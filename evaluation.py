"""
evaluation.py — Accuracy Measurement System

Design Decision: The Problem with "Accuracy" in Generation
    Traditional metrics (Exact Match, BLEU) fail for email replies. 
    If the gold standard is "Sure, 4pm works", and the model generates "That's fine, see you at 16:00", 
    the exact match is 0%, but the semantic accuracy is 100%.

    Our Solution: A Hybrid Metric Approach
    We combine deterministic NLP metrics with an LLM-as-a-judge:
    1. ROUGE-L: Measures lexical overlap and longest common subsequence (sanity check).
    2. Semantic Similarity (SBERT): Measures meaning overlap using cosine similarity of embeddings.
    3. LLM-as-a-Judge: Grades Relevance, Fluency, and Tone, providing reasoning ("the why").
"""

import json
from groq import Groq
import numpy as np
from sentence_transformers import SentenceTransformer, util
from rouge_score import rouge_scorer
import config

class ResponseEvaluator:
    def __init__(self):
        self.groq_client = None
        if config.GROQ_API_KEY:
            try:
                self.groq_client = Groq(api_key=config.GROQ_API_KEY)
            except Exception:
                pass
        
        # Load embedding model for semantic similarity
        try:
            self.embedder = SentenceTransformer(config.EMBED_MODEL)
        except Exception as e:
            self.embedder = None
            print(f"Warning: Could not load embedder for evaluation: {e}")
            
        self.rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    def evaluate(self, incoming_email: str, generated_reply: str, reference_reply: str = None) -> dict:
        """
        Grades the response across multiple dimensions.
        Returns a dictionary containing scores (1-10), reasoning, and a composite score.
        """
        if not self.groq_client:
            return {"error": "Groq client unavailable for evaluation."}

        # 1. Deterministic Metrics (if we have a reference reply from the dataset)
        semantic_sim_score = 0.0
        rouge_l_score = 0.0
        
        if reference_reply and self.embedder:
            # Semantic Similarity (Cosine Similarity between 0 and 10)
            emb1 = self.embedder.encode(generated_reply, convert_to_tensor=True)
            emb2 = self.embedder.encode(reference_reply, convert_to_tensor=True)
            sim = util.cos_sim(emb1, emb2).item()
            semantic_sim_score = max(0.0, sim * 10) # Scale to 0-10
            
            # ROUGE-L (Scale to 0-10)
            rouge_l_score = self.rouge.score(reference_reply, generated_reply)['rougeL'].fmeasure * 10

        # 2. LLM-as-a-Judge Metrics
        system_prompt = """You are an objective AI evaluator.
Grade the quality of an AI-generated email reply based on the incoming email.
Evaluate the response across three dimensions on a scale of 1 to 10:

1. Relevance (1-10): Does it directly answer/address the incoming email's core points? (Also known as Answer Relevance in RAG)
2. Fluency (1-10): Is the text natural, professional, and grammatically sound?
3. Tone (1-10): Is the tone appropriately matched to a corporate setting?

Output strictly valid JSON:
{
    "relevance": {"score": 0.0, "reason": "why"},
    "fluency": {"score": 0.0, "reason": "why"},
    "tone": {"score": 0.0, "reason": "why"}
}"""

        user_prompt = f"Incoming Email:\n{incoming_email}\n\nGenerated Reply:\n{generated_reply}"

        try:
            response = self.groq_client.chat.completions.create(
                model=config.GROQ_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1, # Low temp for deterministic grading
                response_format={"type": "json_object"}
            )
            
            raw_json = response.choices[0].message.content
            parsed = json.loads(raw_json)
            
            # Incorporate deterministic metrics into the result
            parsed["semantic_similarity"] = round(semantic_sim_score, 2) if reference_reply else "N/A"
            parsed["rouge_l"] = round(rouge_l_score, 2) if reference_reply else "N/A"
            
            # Calculate composite score based on LLM dimensions
            w_rel = config.EVAL_WEIGHTS.get('relevance', 0.15)
            w_flu = config.EVAL_WEIGHTS.get('fluency', 0.10)
            w_ton = config.EVAL_WEIGHTS.get('tone', 0.10)
            total_w = w_rel + w_flu + w_ton
            
            s_rel = float(parsed.get("relevance", {}).get("score", 0))
            s_flu = float(parsed.get("fluency", {}).get("score", 0))
            s_ton = float(parsed.get("tone", {}).get("score", 0))
            
            composite = ((s_rel * w_rel) + (s_flu * w_flu) + (s_ton * w_ton)) / total_w
            parsed["composite_score"] = round(composite, 2)
            
            return parsed

        except Exception as e:
            return {"error": f"Evaluation failed: {str(e)}"}

# Singleton
_evaluator = None
def get_evaluator():
    global _evaluator
    if _evaluator is None:
        _evaluator = ResponseEvaluator()
    return _evaluator
