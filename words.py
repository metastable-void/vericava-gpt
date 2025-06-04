from janome.tokenizer import Tokenizer
from janome.analyzer import Analyzer
from janome.charfilter import *
import pandas as pd
import json

t = Tokenizer()
char_filters = [UnicodeNormalizeCharFilter()]
analyzer = Analyzer()

words = []

file_path = "combined-rand.jsonl"
texts = pd.read_json(path_or_buf=file_path, lines=True)
for text in texts["text"]:
    for t in analyzer.analyze(text):
        words.append(t.surface)

print(json.dumps(words))
