"""
train_bpe.py
"""

import argparse
import heapq
import os
import pickle
from collections import defaultdict, Counter
from pathlib import Path
from tqdm import tqdm
from typing import DefaultDict, Dict, Iterable, List, Tuple
from cs336_basics.pretokenization import parallel_pretokenize

# Type aliases
Token = bytes
Pair = Tuple[Token, Token]
Word = Tuple[Token, ...]

def apply_merge(word_tokens: Word, merge_pair: Pair) -> Word:
    """Apply a single merge."""
    a, b = merge_pair
    merged = a + b
    
    new_tokens = []
    i = 0
    while i < len(word_tokens):
        if i < len(word_tokens) - 1 and word_tokens[i] == a and word_tokens[i+1] == b:
            new_tokens.append(merged)
            i += 2
        else:
            new_tokens.append(word_tokens[i])
            i += 1
    return tuple(new_tokens)

def update_counts(
    word_counts: Dict[Word, int],
    pair_counts: Counter[Pair],
    merge_pair: Pair,
) -> Tuple[Dict[Word, int], Counter[Pair]]:
    """Incrementally update word and pair counts after a merge."""
    new_word_counts = {}
    new_pair_counts = Counter(pair_counts)
    
    for word_tokens, freq in word_counts.items():
        if len(word_tokens) < 2:
            new_word_counts[word_tokens] = freq
            continue
        
        old_pairs = list(zip(word_tokens[:-1], word_tokens[1:]))
        if merge_pair not in old_pairs:
            new_word_counts[word_tokens] = freq
            continue
        
        new_word = apply_merge(word_tokens, merge_pair)
        new_word_counts[new_word] = freq
        
        for pair in old_pairs:
            new_pair_counts[pair] -= freq
            if new_pair_counts[pair] == 0:
                del new_pair_counts[pair]
        
        new_pairs = list(zip(new_word[:-1], new_word[1:]))
        for pair in new_pairs:
            new_pair_counts[pair] += freq
    
    return new_word_counts, new_pair_counts

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str] | None = None,
    num_processes: int | None = None,
) -> Tuple[Dict[int, Token], List[Pair]]:
    if vocab_size <= 256:
        raise ValueError("Vocab size must be greater than 256")
    
    special_tokens = special_tokens or ["<|endoftext|>"]
    num_processes = num_processes or min(10, os.cpu_count())
    
    print("[BPE] Pretokenizing corpus...")
    word_counts = parallel_pretokenize(input_path, num_processes, special_tokens)
    
    word_counts = dict(word_counts)
    
    vocab: Dict[int, Token] = {i: bytes([i]) for i in range(256)}
    for tok in special_tokens:
        vocab[len(vocab)] = tok.encode("utf-8")
    
    pair_counts = Counter()
    for word_tokens, freq in word_counts.items():
        if len(word_tokens) < 2:
            continue
        for pair in zip(word_tokens[:-1], word_tokens[1:]):
            pair_counts[pair] += freq
    
    merges: List[Pair] = []
    n_merges = vocab_size - len(vocab)
    
    for _ in tqdm(range(n_merges), desc="Merging pairs"):
        if not pair_counts:
            break
        
        max_pair = max(pair_counts.items(), key=lambda x: (x[1], x[0]))[0]
        
        merged_token = max_pair[0] + max_pair[1]
        vocab[len(vocab)] = merged_token
        merges.append(max_pair)
        
        word_counts, pair_counts = update_counts(word_counts, pair_counts, max_pair)

    return vocab, merges

def _save_pickle(obj, path: str | os.PathLike) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train a byte-level BPE tokenizer.")
    parser.add_argument("--input_path", type=str, required=True, help="Plain-text corpus (UTF-8 encoded)")
    parser.add_argument("--vocab_size", type=int, required=True, help="Target vocabulary size")
    parser.add_argument("--output_path", type=str, required=True, help="Pickle out (vocab, merges)")
    parser.add_argument("--num_processes", type=int, default=os.cpu_count(), help="Parallel workers for pretokenization")
    parser.add_argument("--special_tokens", nargs='+', default=["<|endoftext|>"], help="Special tokens to include in the vocabulary")
    args = parser.parse_args(argv)
    
    vocab, merges = train_bpe(
        args.input_path,
        args.vocab_size,
        args.special_tokens,
        min(10, args.num_processes),
    )
    print(f"[BPE] Vocab size: {len(vocab)}")
    print(f"[BPE] Merges: {len(merges)}")
    
    _save_pickle((vocab, merges), args.output_path)
    print(f"[BPE] Output saved to {args.output_path}")
    
if __name__ == "__main__":
    main()