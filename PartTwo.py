import nltk
import spacy
from pathlib import Path


nlp = spacy.load("en_core_web_sm")
nlp.max_length = 2000000