import os
import pickle
from collections import Counter, defaultdict
from collections.abc import Iterator
from itertools import pairwise
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import TypeAlias

import regex as re
import typer
from tqdm import tqdm

try:
    from .pretokenization_example import find_chunk_boundaries
except ImportError:
    from pretokenization_example import find_chunk_boundaries

app = typer.Typer()

Token: TypeAlias = tuple[bytes, ...]
TokenCounts: TypeAlias = Counter[Token]
NUM_PROCESSES = cpu_count() - 1


class Pretokenizer:
    def __init__(self):
        self.PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        self.BYTE_CACHE = {i: bytes([i]) for i in range(256)}

    def __call__(self, doc: str) -> dict[Token, int]:
        return self.pretokenize(doc)

    def pretokenize(self, doc: str) -> dict[Token, int]:
        pre_tokenised = re.finditer(self.PAT, doc)
        pre_tokenized_dict = defaultdict(int)
        for token in pre_tokenised:
            key = self._string2tuple_of_bytes(token[0])
            pre_tokenized_dict[key] += 1
        return pre_tokenized_dict

    def _string2tuple_of_bytes(self, s: str) -> Token:
        return tuple(self.BYTE_CACHE[b] for b in s.encode("utf-8"))


class ParallelPretokenizer:
    def __init__(self):
        self.pretokenizer = Pretokenizer()

    def __call__(self, chunks: list[str]) -> TokenCounts:
        pretokenised_chunks = self._parallel_pretokenise_chunks(chunks)
        return self._merge_pretokenised_chunks(pretokenised_chunks)

    def _merge_pretokenised_chunks(
        self,
        pretokenised_chunks: Iterator[list[dict[Token, int]]],
    ) -> TokenCounts:
        pretokenized_merged = Counter()
        for chunk_dicts in tqdm(pretokenised_chunks, desc="Merging pretokenized docs"):
            for d in chunk_dicts:
                pretokenized_merged.update(d)
        return pretokenized_merged

    def _parallel_pretokenise_chunks(self, chunks: list[str]) -> Iterator[list[dict[Token, int]]]:
        with Pool(NUM_PROCESSES) as p:
            yield from tqdm(
                p.imap_unordered(self._pretokenise_chunks, chunks, chunksize=1),
                total=len(chunks),
                desc="Pretokenising chunks",
            )

    def _pretokenise_chunks(self, chunk: str) -> list[dict[Token, int]]:
        docs = [d for d in chunk.split("<|endoftext|>") if len(d) > 0]
        return [self.pretokenizer(d) for d in docs]


def consecutive_merging(token2counts: dict[Token, int], num_merges: int) -> list[Token]:
    list_merged_bytes = []
    for _ in tqdm(range(num_merges), desc="Merging bytes"):
        token2counts, merged_bytes = merge_bytes(token2counts)
        list_merged_bytes.append(merged_bytes)
    return list_merged_bytes


def merge_bytes(token2counts: dict[Token, int]) -> dict[Token, int]:
    bytepair_counts = token_count_to_bpe(token2counts)
    to_merge = most_frequent_conseq_token(bytepair_counts)
    token2counts = update_token2counts(token2counts, to_merge)
    return token2counts, to_merge


def token_count_to_bpe(token2counts: dict[Token, int]) -> dict[Token, int]:
    conseq_token_counts = defaultdict(int)
    for token, freq in token2counts.items():
        for pair in pairwise(token):
            conseq_token_counts[pair] += freq

    return conseq_token_counts


def most_frequent_conseq_token(conseq_token_counts: dict[Token, int]) -> Token:
    max_token_counts = max(conseq_token_counts.values())
    # Find lexicographically largest pair of bytes if frequencies tie
    most_frequent_conseq_token = (k for k, v in conseq_token_counts.items() if v == max_token_counts)
    return max(most_frequent_conseq_token)


def update_token2counts(token2counts: dict[Token, int], to_merge: Token):
    # Update pre-tokenized dictionary
    for token in list(token2counts):
        # Iterate over pairwise token
        if to_merge[0] in token and to_merge[1] in token:
            new_key = merge_in_new_token(token, to_merge)
            freq = token2counts.pop(token)
            token2counts[new_key] = freq
    return token2counts


def merge_in_new_token(token: Token, to_merge: Token):
    out, i = [], 0
    while i < len(token):
        if (i + 1 < len(token)) and ((token[i], token[i + 1]) == to_merge):
            out.append(b"".join(to_merge))
            i += 2
        else:
            out.append(token[i])
            i += 1
    return tuple(out)


def _init_vocab(special_tokens: list[str]) -> dict[int, bytes]:
    vocab = {i: bytes([i]) for i in range(256)}
    for i, special_token in enumerate(special_tokens):
        vocab[256 + i] = special_token.encode("utf-8")
    return vocab


def _read_input_in_chunks(input_path: str | os.PathLike) -> list[str]:
    chunks = []
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, NUM_PROCESSES, b"<|endoftext|>")
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            chunks.append(chunk)
    return chunks


def _run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, Token], list[Token]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, Token], list[Token]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    vocab = _init_vocab(special_tokens)
    chunks = _read_input_in_chunks(input_path)
    parallel_pretokenizer = ParallelPretokenizer()
    token2counts = parallel_pretokenizer(chunks)

    new_tokens = consecutive_merging(token2counts, vocab_size - 256 - len(special_tokens))

    vocab_len = len(vocab)
    for i, new_token in enumerate(new_tokens):
        vocab[vocab_len + i] = b"".join(new_token)
    return vocab, new_tokens


@app.command()
def train_bpe(
    input_path: Path,
    vocab_size: int,
    special_tokens: list[str],
):
    vocab, merges = _run_train_bpe(input_path, vocab_size, special_tokens)
    vocab_path = input_path.with_suffix(".pkl").with_stem("vocab_" + input_path.stem)
    merges_path = input_path.with_suffix(".pkl").with_stem("merges_" + input_path.stem)
    with open(vocab_path, "wb") as f:
        pickle.dump(vocab, f)
    with open(merges_path, "wb") as f:
        pickle.dump(merges, f)
    return vocab, merges


if __name__ == "__main__":
    app()
