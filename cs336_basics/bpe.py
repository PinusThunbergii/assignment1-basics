import os
from typing import BinaryIO, Dict, Generic, Iterable, List, Optional, Set, Tuple
from multiprocessing import Pool
from tqdm import tqdm
import time
import regex as re
from collections import Counter
import cProfile
import json
# from utils import pre_tokenize_single_chunk


def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]] : 
    
    vocab = list()
    vocab = [ bytes([i]) for i in range(256) ] + [i.encode("utf-8") for i in special_tokens]
    
    num_processes = 10
    
    corpus = pre_tokenize(input_path, num_processes, special_tokens)

    vocab, merges = create_merges(corpus, vocab, vocab_size)

    return vocab, merges



'''
Реалистичные уровни ускорения:

  1. Микрооптимизация текущего кода
     Убрать bfind_all, slicing, i not in find_pos, заменить на один линейный проход.

     Ожидаемо: 2-5x.

  2. Параллелизация полного пересчета
     Раскидать подсчет пар и применение merge по процессам.

     Ожидаемо: 2-8x на большом корпусе, но на маленьком тесте может быть медленнее из-за pickle/IPC. Алгоритмически всё равно плохо.

  3. Инкрементальный BPE
     Не пересчитывать все пары каждый раз. Хранить pair_counts, pair -> affected_words, heap для max pair, и после merge обновлять только затронутые слова.

     Ожидаемо: 10-100x+ на большом корпусе. Это главный выигрыш.

  4. Оптимизация представления данных
     Использовать integer token ids вместо bytes объектов в tuple.

     Ожидаемо: еще 2-4x сверху, иногда больше, потому что bytes tuples дороги по памяти и hashing.

  В сумме: текущий create_merges можно улучшить не на проценты, а на один-два порядка для больших данных. Самая важная мысль: параллелить текущий полный пересчет можно, но это лечит симптом.
  Основная проблема в алгоритме: после каждого merge меняется только малая часть corpus, а код пересчитывает всё.
'''


'''


push new version into heap
old version remains in heap
when popped:
    compare heap count with current pair_counts[pair]
    if stale, skip

initial scan once:
    count all pairs
    remember where each pair occurs

for each merge:
    choose best pair from heap
    find only words containing this pair
    update pair counts only for those words
    
'''

import heapq


def pop_max_valid(max_heap, pair_counts) -> Tuple[bytes, bytes]:
    while len(max_heap) != 0:
        count, pair = heapq.heappop_max(max_heap)

        current_count = pair_counts.get(pair, 0)
        # смотрим не устарела ли пара, если да идем за следующей
        if current_count > 0 and current_count == count:
            return pair

    else:
        raise ValueError("TODO")

def create_merges(corpus: Counter[tuple[bytes,...]], vocab: list[bytes], vocab_size: int) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    words = [item for item, count in corpus.items()]
    word_counts = [count for item, count in corpus.items()]
    pair_counts, pair_to_words = make_pair_and_counts(corpus)
    max_heap = [(count, pair) for pair, count in pair_counts.items()]
    heapq.heapify_max(max_heap)

    merges = list()
    for i in tqdm(range(len(vocab), vocab_size), desc="merging"):
        merge = pop_max_valid(max_heap, pair_counts)
        affected_word_ids = set(pair_to_words[merge])
        new_token = b''.join(merge)

        # update
        for word_id in affected_word_ids:
            old_word = words[word_id]
            word_count = word_counts[word_id]

            old_adj_pairs = make_pairs(old_word)
            old_unique_pairs = set(old_adj_pairs)

            # pair_counts обновляем по всем occurrences, включая дубликаты
            for old_adj_pair in old_adj_pairs:
                pair_counts[old_adj_pair] -= word_count

            # pair_to_words обновляем только по уникальным парам
            for old_adj_pair in old_unique_pairs:
                pair_to_words[old_adj_pair].discard(word_id)

                if pair_counts.get(old_adj_pair, 0) <= 0:
                    pair_counts.pop(old_adj_pair, None)
                    pair_to_words.pop(old_adj_pair, None)
                else:
                    heapq.heappush_max(max_heap, (pair_counts[old_adj_pair], old_adj_pair))

            new_word: List[bytes] = list()
            i = 0

            while i < len(old_word):
                if i + 1 < len(old_word) and old_word[i] == merge[0] and old_word[i + 1] == merge[1]:
                    new_word.append(new_token)
                    i += 2
                else:
                    new_word.append(old_word[i])
                    i += 1

            new_word = tuple(new_word)
            words[word_id] = new_word
            
            new_adj_pairs = make_pairs(new_word)
            new_unique_pairs = set(new_adj_pairs)

            # pair_counts обновляем по всем occurrences
            for new_adj_pair in new_adj_pairs:
                pair_counts[new_adj_pair] += word_count

            # pair_to_words обновляем только по уникальным парам
            for new_adj_pair in new_unique_pairs:
                pair_to_word = pair_to_words.get(new_adj_pair, set())
                pair_to_word.add(word_id)
                pair_to_words[new_adj_pair] = pair_to_word

                heapq.heappush_max(max_heap, (pair_counts[new_adj_pair], new_adj_pair))

        vocab.append(new_token)
        merges.append(merge)
    new_vocab = { i:v for i, v in enumerate(vocab)}
    return new_vocab, merges

def update_corpus(corpus: Counter[tuple[bytes]], pair_to_merge: bytes, joined_merge: bytes):
    new_corpus = Counter()
        
    for k, v in corpus.items():
        finds_pos = find_all_idx_to_update(k, pair_to_merge)
        if len(finds_pos) == 0:
            new_corpus[k] = v
            continue
            
        i = 0
        new_corpus_item = []
        while(i < len(k)):
            if i not in finds_pos:
                new_corpus_item.append(k[i])
                i += 1
            else:
                new_corpus_item.append(joined_merge)
                i += len(pair_to_merge)
        new_corpus[tuple(new_corpus_item)] = v
    return new_corpus


def make_pairs(word: Tuple[bytes,...]) -> List[Tuple[bytes, bytes]]:
    
    pairs: List[Tuple[bytes, bytes]] = list()

    
    for a, b in zip(word, word[1:]):
        pair = (a, b)
        pairs.append(pair)

    return pairs
    
    
    return

def make_pair_and_counts(corpus) -> Tuple[Counter[Tuple[bytes, bytes]], Dict[Tuple[bytes, bytes], Set[int]]]: 
    pair_counts = Counter()
    pair_to_words: Dict[Tuple[bytes, bytes], Set[int]] = dict()


    for i, (word, count) in enumerate(corpus.items()):

        for a, b in zip(word, word[1:]):
            pair = (a, b)
            pair_counts[pair] += count

            words_ids = pair_to_words.get(pair, set())
            words_ids.add(i)
            pair_to_words[pair] = words_ids

    return pair_counts, pair_to_words

def find_all_idx_to_update(x: list[bytes], sub: list[bytes]) -> list[int]:
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

def make_chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def pre_tokenize(input_path: str, num_processes: int, special_tokens: list[str]) -> Counter[tuple[bytes]]:
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, 200, "<|endoftext|>".encode("utf-8"))
    
        # The following is a serial implementation, but you can parallelize this 
        # by sending each start/end pair to a set of processes.
        merge_counters = Counter()
        # chunks_pos = list(zip(boundaries[:-1], boundaries[1:]))
        chunks_pos = list(make_chunks(list(zip(boundaries[:-1], boundaries[1:])), 28))
         
        # for chunk_pos in tqdm(make_chunks(chunks_pos, 28)):
        for chunk_pos in tqdm(chunks_pos, desc="pretokenization"):
            chunks = []  
            for start, end in chunk_pos:
                f.seek(start)
                chunk = f.read(end - start).decode("utf-8", errors="ignore")
                # Run pre-tokenization on your chunk and store the counts for each pre-token
                chunks.append(chunk)
                
            with Pool(28) as pool:
                pretoken_counters = list(pool.starmap(pre_tokenize_single_chunk, [ (chunk, special_tokens) for chunk in chunks]))
                        # pretokenize_chunks = pool.starmap(pre_tokenize_single_chunk, [ (chunk, special_tokens) for chunk in chunks])
            pretoken_counters.append(merge_counters)
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


def save_to_json(object, path):
    with open(path, "w", encoding="utf-8") as f:
        # json.dump(object, f, ensure_ascii=False)
        json.dump(object, f)

def main():
    import datetime
    start = datetime.datetime.now()
    # vocab, merges = train_bpe("./data/TinyStoriesV2-GPT4-valid.txt", 1000, ["<|endoftext|>"])
    # vocab, merges = train_bpe("./data/TinyStoriesV2-GPT4-train.txt", 10000, ["<|endoftext|>"])
    
    vocab, merges = train_bpe("./data/owt_train.txt", 32000, ["<|endoftext|>"])

    vocab = { v:k.decode("utf-8", errors="replace") for v, k in vocab.items()}
    merges = [ (a.decode("utf-8", errors="replace"), b.decode("utf-8", errors="replace")) for a, b in merges]
    stop = datetime.datetime.now()
    elapsed = (stop - start).seconds
    print(f"Elapsed {elapsed}s")
    save_to_json(vocab, "vocab_owt_train.json")
    save_to_json(merges, "merges_owt_train.json")
    # train_bpe("./data/TinyStoriesV2-GPT4-valid.txt", 1000, ["<|endoftext|>"])
    # train_bpe("./data/owt_train.txt", 1000, ["<|endoftext|>"])
    return

if __name__ == '__main__':
    # cProfile.run('main()')
    main()
    
    
# scalene --html --output foo.html cs336_basics/bpe.py
# deactivate
# conda activate base
# uv run pytest tests/test_train_bpe.py
# scalene --cpu --profile-all --html --output foo.html cs336_basics/bpe.py 
# scalene  --profile-all --html --output foo.html cs336_basics/bpe.py 
# python -m cProfile -s cumulative  cs336_basics/bpe.py
# python cs336_basics/bpe.py 
# source ./.venv/bin/activate
# viztracer cs336_basics/bpe.py 

# uv run pytest tests/test_tokenizer.py