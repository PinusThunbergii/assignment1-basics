import os
from typing import BinaryIO, Optional
from multiprocessing import Pool
from tqdm import tqdm
import time
import regex as re
from collections import Counter
import cProfile
from utils import pre_tokenize_single_chunk


def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]] : 
    
    # vocab = { i:bytes([i]) for i in range(256) }
    vocab = list()
    vocab = [ bytes([i]) for i in range(256) ] + [i.encode("utf-8") for i in special_tokens]
    
    num_processes = 10
    
    corpus = pre_tokenize(input_path, num_processes, special_tokens)
    
    merges = list()

    c = Counter()
    
    for k, v in corpus.items():
        # t = tuple([bytes([x]) for x in list(k)])
        for a, b in zip(k, k[1:]):
            c[(a, b)] += v
    
    merge = get_max(c)
    joined_merge = b''.join(merge)
    vocab.append(joined_merge)
    merges.append((merge[0], merge[1]))
    
    new_corpus = Counter()
    
    for k, v in corpus.items():
        
        find_pos = bfind_all(k, merge)
        if len(find_pos) == 0:
            new_corpus[k] = v
            continue
        # a, b, c, d, b, c, e => a, bc, d, bc, e pos [1, 4]
        
        i = 0
        new_k = []
        while(i < len(k)):
            if i not in find_pos:
                new_k.append(k[i])
                i += 1
            else:
                new_k.append(joined_merge)
                i += len(merge)
        new_corpus[tuple(new_k)] = v
            
    return vocab, merges

def bfind_all(x: list[bytes], sub: list[bytes]) -> list[int]:
    if len(x) < len(sub):
        return []
    
    finds = []
    pos = -1
    
    for i in range(0, len(x) - len(sub) + 1):
        if sub == x[i:i+len(sub)]:
            pos = i
            finds.append(pos)
        
    return finds

def bfind(x: list[bytes], sub: list[bytes], start: Optional[int] = None) -> int:
    if start is None:
        start = 0
    assert start <= len(x) - len(sub) + 1
    assert len(x) > len(sub)
    
    pos = -1
    
    for i in range(start, len(x) - len(sub) + 1):
        if sub == x[i:i+len(sub)]:
            pos = i
            break
        
    return pos


def pre_tokenize(input_path: str, num_processes: int, special_tokens: list[str]):
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, "<|endoftext|>".encode("utf-8"))
    
        # The following is a serial implementation, but you can parallelize this 
        # by sending each start/end pair to a set of processes.
        chunks = []    
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            # Run pre-tokenization on your chunk and store the counts for each pre-token
            chunks.append(chunk)
                    
    merge_counters = Counter()
            
    with Pool(num_processes) as pool:
        pretoken_counters = list(tqdm(
                pool.starmap(pre_tokenize_single_chunk, [ (chunk, special_tokens) for chunk in chunks])
        ))
            # pretokenize_chunks = pool.starmap(pre_tokenize_single_chunk, [ (chunk, special_tokens) for chunk in chunks])

    merge_counters = merge_pretoken_counters(pretoken_counters)
        
    corpus = Counter()    
        
    for k, v in merge_counters.items():
        t = tuple([bytes([x]) for x in list(k)])
        corpus[t] = v
    # return merge_counters
    return corpus


def get_max(x: Counter[tuple[bytes], int]) -> bytes:
    max_value = max(x.values())
    max_keys = [k for k, v in x.items() if v == max_value]
    return sorted(max_keys, reverse=True)[0]
    

def create_pairs(x: Counter[tuple[bytes], int]) -> Counter[tuple[bytes], int]:
    y = Counter()
    
    for k, v in x:
        for a, b in zip(k, k[1:]):
            y[(a, b)] += v
    
    return y


def merge_pretoken_counters(counters: list[Counter[bytes, int]]):
    assert len(counters) != 0
    if len(counters) == 1:
        return counters[0]
    
    a = counters[0]
    
    for b in counters[1:]:
        a = a + b
    
    return a        

# def pre_tokenize_single_chunk(chunk: str, special_tokens: list[str]) -> Counter[bytes, int] :
#     PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    
#     delimiter_tokens = [re.escape(x) for x in special_tokens]
#     delimiter = "|".join(delimiter_tokens)
    
#     pretoken_counter = Counter()
    
#     for piece in re.splititer(delimiter, chunk):
#        for word in re.finditer(PAT, piece):
#            word = word.group(0).encode("utf-8")
#            pretoken_counter[word] += 1
        
#     return pretoken_counter

def find_chunk_boundaries(
    file: BinaryIO, 
    desired_num_chunks: int, 
    split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def main():
    # train_bpe("./data/TinyStoriesV2-GPT4-train.txt", 1000, ["<|endoftext|>"])
    train_bpe("./data/TinyStoriesV2-GPT4-valid.txt", 1000, ["<|endoftext|>"])
    # train_bpe("./data/owt_train.txt", 1000, ["<|endoftext|>"])
    return

if __name__ == '__main__':
    # cProfile.run('main()')
    main()
    
    
# scalene --html --output foo.html cs336_basics/bpe.py