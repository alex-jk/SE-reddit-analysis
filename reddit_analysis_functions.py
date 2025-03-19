import spacy
from sentence_transformers import SentenceTransformer, util
import torch
import pandas as pd

class ViolenceModel:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        self.violent_phrases = [
            "tried to strangle me",
            "tried to kill me",
            "choked me",
            "assaulted me",
            "strangulation",
            "aggravated assault"
        ]
        
        self.reference_embeddings = self.embed_model.encode(
            self.violent_phrases, 
            convert_to_tensor=True
        )
    
    def embed_text(self, text):
        return self.embed_model.encode(text, convert_to_tensor=True)

    def extract_sliding_phrases(self, text, window_size=4):
        """
        Extracts sliding windows of 'window_size' words from each sentence in the text.
        """
        doc = self.nlp(text)
        phrases = []
        
        for sent in doc.sents:
            words = [token.text for token in sent if not token.is_punct]
            for i in range(len(words) - window_size + 1):
                phrase = ' '.join(words[i:i + window_size])
                phrases.append(phrase)
                
        return phrases

    def find_similar_phrases(self, text, threshold=0.7):
        """
        Extracts sliding phrases and finds those similar to violent reference phrases.
        Returns matched phrases with similarity scores above the threshold.
        """
        matches = []
        phrases = self.extract_sliding_phrases(text)
        
        if not phrases:
            return matches
        
        # Embed all phrases at once
        phrase_embeddings = self.embed_model.encode(phrases, convert_to_tensor=True)
        
        # Compute cosine similarities
        cosine_scores = util.pytorch_cos_sim(phrase_embeddings, self.reference_embeddings)
        
        for idx, phrase in enumerate(phrases):
            max_score = torch.max(cosine_scores[idx]).item()
            
            if max_score >= threshold:
                matches.append((phrase, max_score))
        
        return matches