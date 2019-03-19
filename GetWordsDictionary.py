import numpy as np
import scipy
import pandas as pd
import nltk

def GetWords(sentences) :

    words = []
    for tokens in sentences:
        for word in tokens:
            words.append(word)

    words = set(words)

    words = sorted(words)

    AllWords = list(words)

    return AllWords
