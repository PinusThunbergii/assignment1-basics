

from typing import Iterable, Iterator
import regex as re
import os
import json

class Tokenizer:
    
    def __init__(self, 
                 vocab: dict[int, bytes], 
                 merges: list[tuple[bytes, bytes]], 
                 special_tokens : list[str] | None = None):
        
        self.vocab = vocab
        self.vocab_reverse = { v:k for k, v in vocab.items()}
        self.merges = merges
        self.merges_dict = { m:i for i, m in enumerate(merges)}
        
        self.PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self.pat = re.compile(self.PAT)

        if special_tokens != None:
            self.special_tokens = special_tokens
            delimiter_tokens = [re.escape(x) for x in self.special_tokens]
            delimiter = "|".join(delimiter_tokens)
            self.delimiter = re.compile(delimiter)
        else:
            self.special_tokens = None
            self.delimiter = None
    
        # print(self.vocab_reverse)
        
    @staticmethod
    def from_files(vocab_filepath: str, merges_filepath: str, special_tokens : list[str] | None = None) :
        
        vocab = {}
        merges = []
        with open(vocab_filepath, "r") as f:
            vocab = json.load(f)
            vocab = { int(k):v.encode("utf-8", errors="replace") for k,v in vocab.items()}
        
        with open(merges_filepath, "r") as f:
            merges = json.load(f)
            merges = [ (x[0].encode("utf-8", errors="replace"), x[1].encode("utf-8", errors="replace")) for x in merges]
        
        return Tokenizer(vocab, merges, special_tokens)        
    
    def get_merge_idx(self, pair: tuple[bytes, bytes]) -> int: 
        if pair not in self.merges_dict.keys():
            return -1
        
        idx = self.merges_dict[pair]
        return idx    
    
    def encode(self, text: str) -> list[int]:
        pre_tokens = []
        
        for piece in re.splititer(self.delimiter, text):
            for word in re.finditer(self.pat, piece):
                pre_tokens.append(word.group(0).encode("utf-8", errors="replace"))
                
            
        # for word in re.finditer(self.pat, text):
        #     pre_tokens.append(word.group(0).encode("utf-8", errors="replace"))
        
        output = []
        
        for pre_token in pre_tokens:
            chars = [bytes([x]) for x in pre_token]
            
            while True:
                
                pairs = list(zip(chars, chars[1:]))
                # pairs_idx = [self.get_merge_idx(pair) for pair in pairs]
                
                pairs_idx = []
                
                for pair in pairs:
                    pair_idx = self.get_merge_idx(pair)
                    if pair_idx != -1:
                        pairs_idx.append(pair_idx)
                
                if len(pairs_idx) == 0: # we don't have pairs to merge
                    break
                pair_to_merge_id = min(pairs_idx)
                pair_to_merge = self.merges[pair_to_merge_id]
                
                # apply merge
                tmp_chars = []
                
                i = 0
                while True:
                    if len(chars) == i:
                        break
                    
                    if chars[i] == pair_to_merge[0] and chars[i + 1] == pair_to_merge[1]:
                        tmp_chars.append(pair_to_merge[0] + pair_to_merge[1])
                        i += 2
                    else:
                        tmp_chars.append(chars[i])
                        i += 1

                chars = tmp_chars

            # convert chars to ids
            for char in chars:
                if char in self.vocab_reverse.keys():
                    id = self.vocab_reverse[char]
                    output.append(id)
                else:
                    print(f"Not found {char}")
                   
            # print(output) 
            # list(zip(tmp, tmp[1:]))
            # [self.merges.index(x) for x in pairs]

            
        return output
    

    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        return []
    
    def decode(self, ids: list[int]) -> str:
        output_bytes = bytes()
        
        for id in ids:
            output_bytes += self.vocab[id]
            
        output_str = output_bytes.decode(encoding="utf-8", errors="replace")
        return output_str

if __name__ == "__main__":
    tokenizer = Tokenizer.from_files(vocab_filepath="vocab.json", merges_filepath="merges.json", special_tokens=["<|endoftext|>"])
    # test_text = "Hello world!"
    # test_text = "Hello how <|endoftext|><|endoftext|>  are you? 🙃<|endoftext|>"
    test_text = "Héllò hôw <|endoftext|><|endoftext|> are ü? 🙃<|endoftext|>"
    encoded = tokenizer.encode(test_text)
    print(encoded)
    decoded = tokenizer.decode(encoded)
    print(decoded)
    
# deactivate 
# conda activate base
# uv run pytest tests/test_tokenizer.py