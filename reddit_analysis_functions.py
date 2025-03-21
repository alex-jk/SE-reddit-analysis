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
            "hit and strangled,"
            "tried to kill me",
            "choked me",
            "had been choked",
            "physically assaulted me",
            "strangulation",
            "aggravated assault",
            "beat me",
            "hit me",
            "tried to murder me",
            "physically violent"
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

# Define a helper function for labeling
def label_post_as_violent(text, model, threshold=0.7):
    matches = model.find_similar_phrases(text, threshold=threshold)
    
    # Ensure unique matches
    unique_matches = list(set(matches)) if matches else []
    
    # Label is 1 if there are any matches, otherwise 0
    label = 1 if unique_matches else 0
    
    return label, unique_matches

