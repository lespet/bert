# -*- coding: utf-8 -*-
"""
Created on Sat Aug  7 15:49:05 2021

@author: user
"""

sentences = [
    "Three years later, the coffin was still full of Jello.",
    "The fish dreamed of escaping the fishbowl and into the toilet where he saw his friend go.",
    "The person box was packed with jelly many dozens of months later.",
    "Standing on one's head at job interviews forms a lasting impression.",
    "It took him a month to finish the meal.",
    "He found a leprechaun in his walnut shell."
]
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('bert-base-nli-mean-tokens')
sentence_embeddings = model.encode(sentences)
sentence_embeddings.shape
sentence_embeddings
from sklearn.metrics.pairwise import cosine_similarity
cos_sim=cosine_similarity(
    [sentence_embeddings[0]],
    sentence_embeddings[1:]
)