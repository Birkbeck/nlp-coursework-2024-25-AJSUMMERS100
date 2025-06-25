import nltk
import spacy
from pathlib import Path
import pandas as pd

nlp = spacy.load("en_core_web_sm")
nlp.max_length = 2000000

def part_a(path):
    pre_df = pd.read_csv(path)
    df = pd.DataFrame(pre_df)
    df.replace("Labour (Co-op)", "Labour", inplace = True)
    return df

if __name__ == "__main__":
    path = Path.cwd() / "p2-texts" / "hansard40000.csv"
    
    # Part a
    df = part_a(path)
    print(df.shape)

    # Part b