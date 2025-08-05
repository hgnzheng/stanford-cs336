"""
pretokenization.py

Usage:
python pretokenization.py --input_path <input_path> --output_path <output_path> --num_processes <num_processes> --special_tokens <special_tokens>

Quick Example:
python pretokenization.py --input_path ../data/TinyStoriesV2-GPT4-valid.txt --output_path ../data/pretokenization/TinyStoriesV2-GPT4-valid_pretokenization.pickle
"""


import os
from typing import BinaryIO
import argparse
import multiprocessing as mp
import regex as re
from collections import defaultdict, Counter
import pickle
from tqdm.auto import tqdm
from typing import Iterable

# GPT-2 regex pattern
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


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

    for bi in tqdm(range(1, len(chunk_boundaries) - 1),
                    desc="Finding chunk boundaries"):
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


def process_chunk(args):
    path, start, end, split_pattern, pat_pattern, special_token_set, worker_id = args
    
    with open(path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
    
    
    word_counts = Counter()
    segments = re.split(split_pattern, chunk)
    
    for segment in tqdm(segments,
                        desc=f"Worker {worker_id} Segments", 
                        position=worker_id, 
                        leave=False):
        if not segment:
            continue
        if segment in special_token_set:
            word_bytes = (segment.encode("utf-8"),)
            word_counts[word_bytes] += 1
        else:
            pretokens = re.findall(pat_pattern, segment)
            for word in pretokens:
                word_bytes = tuple(bytes([b]) for b in word.encode("utf-8"))
                word_counts[word_bytes] += 1
    
    return word_counts

def parallel_pretokenize(
    path: str,
    num_processes: int = os.cpu_count(),
    special_tokens: list[str] | None = None
) -> Counter[str]:
    pat_pattern = re.compile(PAT)
    special_token_set = set(special_tokens)
    
    split_pattern_str = f'({"|".join(map(re.escape, special_tokens))})'
    split_pattern = re.compile(split_pattern_str)
    
    with open(path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, special_tokens[0].encode("utf-8"))
        
    jobs = [
        (path, boundaries[i], boundaries[i+1], split_pattern, pat_pattern, special_token_set, i)        for i in range(len(boundaries) - 1)
        if boundaries[i] < boundaries[i+1]
    ]
    
    total_word_counts = Counter()
    
    with mp.Pool(num_processes) as pool:
        for chunk_counts in tqdm(pool.imap_unordered(process_chunk, jobs),
                                    total=len(jobs),
                                    desc="Pretokenizing chunks"):
            total_word_counts.update(chunk_counts)
    print(f"Learned {len(total_word_counts)} tokens")
    return total_word_counts

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--num_processes", type=int, default=os.cpu_count())
    parser.add_argument("--special_tokens", nargs='+', default=["<|endoftext|>"])
    args = parser.parse_args()
    
    total_word_counts = parallel_pretokenize(
        args.input_path,
        args.num_processes,
        args.special_tokens
    )
    
    with open(args.output_path, "wb") as f:
        pickle.dump(total_word_counts, f)
    print(f"Word counts saved to {args.output_path}")
    
if __name__ == "__main__":
    main()