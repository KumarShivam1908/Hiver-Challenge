"""
app.py — Presentation Layer (Streamlit UI)

Design Decision:
    Streamlit allows us to rapidly prototype the UI while maintaining a clean Pythonic backend.
    We split the UI into clear conceptual tabs:
    1. Generator: For inference testing (putting a new email in and getting a response).
    2. Data: Transparency into the retrieval context.
    3. Accuracy: Explaining the evaluation framework (crucial for the challenge).
"""

import streamlit as st
import pandas as pd
import config
from generator import get_generator
from evaluation import get_evaluator

st.set_page_config(page_title="Email GenAI Agent", layout="wide")

def main():
    st.title("GenAI Email Reply Agent")
    st.markdown("End-to-end RAG system for intelligent email drafting.")
    
    if not config.GROQ_API_KEY:
        st.error("Missing GROQ_API_KEY in `.env`.")

    tab1, tab2, tab3 = st.tabs(["Inference & Testing", "Data & Retrieval", "Accuracy System"])

    # ─── TAB 1: INFERENCE ──────────────────────────────────────────────────
    with tab1:
        st.header("Response Generator")
        col1, col2 = st.columns(2)
        
        with col1:
            incoming = st.text_area("Incoming Email", height=200)
            reference = st.text_area("Expected Ground-Truth Reply (Optional)", height=100, 
                                     help="Required to calculate SBERT & ROUGE-L. The LLM Judge works without it!")
            
            if st.button("Generate Reply", type="primary"):
                if incoming.strip():
                    with st.spinner("Generating..."):
                        gen = get_generator()
                        reply, examples = gen.generate_reply(incoming)
                        
                        st.session_state['incoming'] = incoming
                        st.session_state['reference'] = reference if reference.strip() else None
                        st.session_state['reply'] = reply
                        st.session_state['examples'] = examples
                        st.session_state['eval'] = None
                else:
                    st.warning("Please enter an email.")

        with col2:
            if 'reply' in st.session_state:
                st.subheader("Suggested Reply")
                st.info(st.session_state['reply'])
                
                st.divider()
                st.subheader("Accuracy Evaluation")
                if st.button("Run Evaluation"):
                    with st.spinner("Grading..."):
                        evaluator = get_evaluator()
                        res = evaluator.evaluate(
                            incoming_email=st.session_state['incoming'], 
                            generated_reply=st.session_state['reply'],
                            reference_reply=st.session_state.get('reference')
                        )
                        st.session_state['eval'] = res
                
                if st.session_state.get('eval'):
                    e = st.session_state['eval']
                    if "error" in e:
                        st.error(e["error"])
                    else:
                        st.metric("Composite LLM Score", f"{e.get('composite_score', 0)} / 10")
                        
                        ec1, ec2, ec3 = st.columns(3)
                        ec1.metric("Relevance (LLM)", f"{e.get('relevance', {}).get('score', 0)}")
                        ec2.metric("Fluency (LLM)", f"{e.get('fluency', {}).get('score', 0)}")
                        ec3.metric("Tone (LLM)", f"{e.get('tone', {}).get('score', 0)}")
                        
                        ec4, ec5 = st.columns(2)
                        ec4.metric("Semantic Similarity (SBERT)", str(e.get("semantic_similarity", "N/A")))
                        ec5.metric("ROUGE-L", str(e.get("rouge_l", "N/A")))
                        
                        with st.expander("View LLM Grading Reasoning", expanded=True):
                            st.markdown(f"**Relevance:** {e.get('relevance', {}).get('reason', '')}")
                            st.markdown(f"**Fluency:** {e.get('fluency', {}).get('reason', '')}")
                            st.markdown(f"**Tone:** {e.get('tone', {}).get('reason', '')}")

    # ─── TAB 2: DATA ───────────────────────────────────────────────────────
    with tab2:
        st.header("Data & Context")
        if 'examples' in st.session_state and st.session_state['examples']:
            st.subheader("Retrieved Few-Shot Examples")
            for i, ex in enumerate(st.session_state['examples']):
                with st.expander(f"Context Example {i+1}"):
                    st.markdown(f"**Incoming:** {ex['incoming']}")
                    st.markdown(f"**Reply:** {ex['reply']}")
        else:
            st.info("Generate a reply first to see retrieved context.")
            
        st.divider()
        st.subheader("Raw Dataset Viewer")
        try:
            df = pd.read_csv(config.DATA_CSV, nrows=100)
            st.dataframe(df[['EmailSend', 'EmailReply']], use_container_width=True)
            st.caption("Showing 100 rows from the local dataset.")
        except Exception:
            st.warning("Could not load local dataset for viewing.")

    # ─── TAB 3: ACCURACY SYSTEM ───────────────────────────────────────────
    with tab3:
        st.header("Accuracy Measurement System")
        st.markdown("""
        ### Why Not Exact Match?
        Exact Match or BLEU scores grade valid generative replies near 0%. We utilize an LLM-as-a-Judge framework to evaluate the output on three continuous dimensions:
        
        1. **Relevance (1-10):** Does the text solve the user's intent?
        2. **Fluency (1-10):** Is the text grammatically sound?
        3. **Tone (1-10):** Does it match corporate (Enron) standards?
        
        This provides a numeric metric that reflects reality and gives the reasoning behind the score.
        """)

if __name__ == "__main__":
    main()
