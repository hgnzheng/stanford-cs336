import regex as re
from typing import Dict, List, Tuple, Iterator, Iterable, Optional
import pickle

# GPT-2 regex pattern
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class Tokenizer:
    def __init__(
        self,
        vocab: Dict[int, bytes],
        merges: List[Tuple[bytes, bytes]],
        special_tokens: Optional[List[str]] = None,
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        
        self.token2id = {token: token_id for token_id, token in self.vocab.items()}
        
        for special_token in self.special_tokens:
            special_token_bytes = special_token.encode("utf-8")
            if special_token_bytes not in self.token2id:
                new_id = len(self.vocab)
                self.vocab[new_id] = special_token_bytes
                self.token2id[special_token_bytes] = new_id
        
        self.pat_pattern = re.compile(PAT)
        
        if self.special_tokens:
            sorted_tokens = sorted(self.special_tokens, key=len, reverse=True)
            split_pattern_str = f'({"|".join(map(re.escape, sorted_tokens))})'
            self.split_pattern = re.compile(split_pattern_str)
        else:
            self.split_pattern = None
        
        self.special_token_set = set(self.special_tokens)
    
    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: Optional[List[str]] = None,
    ):
        with open(vocab_filepath, 'rb') as f:
            vocab = pickle.load(f)
        with open(merges_filepath, 'rb') as f:
            merges = pickle.load(f)
        return cls(vocab, merges, special_tokens)
    
    def _apply_merges(self, word_tokens: Tuple[bytes, ...]) -> Tuple[bytes, ...]:
        """Apply a single merge."""
        if len(word_tokens) < 2:
            return word_tokens
        
        curr_tokens = list(word_tokens)
        
        for merge_pair in self.merges:
            if len(curr_tokens) < 2:
                break
            
            i = 0
            new_tokens = []
            while i < len(curr_tokens):
                if i < len(curr_tokens) - 1 and curr_tokens[i] == merge_pair[0] and curr_tokens[i+1] == merge_pair[1]:
                    new_tokens.append(merge_pair[0] + merge_pair[1])
                    i += 2
                else:
                    new_tokens.append(curr_tokens[i])
                    i += 1
            curr_tokens = new_tokens
        return tuple(curr_tokens)
    
    def encode(self, text: str) -> List[int]:
        """Encode text into a list of token IDs."""
        if not text:
            return []
        
        ids = []
        if self.split_pattern and self.special_tokens:
            segments = self.split_pattern.split(text)
        else:
            segments = [text]
        
        for segment in segments:
            if not segment:
                continue
            
            if segment in self.special_token_set:
                special_token_bytes = segment.encode("utf-8")
                if special_token_bytes in self.token2id:
                    ids.append(self.token2id[special_token_bytes])
            else:
                pretokens = self.pat_pattern.findall(segment)
                for pretoken in pretokens:
                    word_bytes = tuple(bytes([b]) for b in pretoken.encode("utf-8"))
                    
                    merged_tokens = self._apply_merges(word_bytes)
                    
                    for token in merged_tokens:
                        if token in self.token2id:
                            ids.append(self.token2id[token])
        
        return ids
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Lazily encode an iterable of strings."""
        for text in iterable:
            for token_id in self.encode(text):
                yield token_id
    
    def decode(self, ids: List[int]) -> str:
        """Decode token ids back to text."""
        bytes_list = []
        for token_id in ids:
            if token_id in self.vocab:
                bytes_list.append(self.vocab[token_id])
        
        all_bytes = b"".join(bytes_list)
        return all_bytes.decode("utf-8", errors='replace')