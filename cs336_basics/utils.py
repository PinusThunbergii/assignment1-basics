import os
import regex as re
from collections import Counter

def pre_tokenize_single_chunk(chunk: str, special_tokens: list[str]) -> Counter[bytes, int] :
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    
    pat = re.compile(PAT)
    
    delimiter_tokens = [re.escape(x) for x in special_tokens]
    delimiter = "|".join(delimiter_tokens)
    delimiter = re.compile(delimiter)
    pretoken_counter = Counter()
    
    for piece in re.splititer(delimiter, chunk):
       for word in re.finditer(pat, piece):
        #    word = word.group(0).encode("utf-8")
           word = word.group(0).encode("utf-8")
           pretoken_counter[word] += 1
        
    return pretoken_counter