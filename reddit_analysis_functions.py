import spacy
from sentence_transformers import SentenceTransformer, util
import torch
import pandas as pd

# Load models
nlp = spacy.load('en_core_web_sm')
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# Violent reference phrases
violent_reference_phrases = [
    "tried to strangle me",
    "tried to kill me",
    "choked me",
    "assaulted me",
    "strangulation",
    "aggravated assault"
]

# Precompute embeddings for reference phrases
reference_embeddings = embed_model.encode(violent_reference_phrases, convert_to_tensor=True)