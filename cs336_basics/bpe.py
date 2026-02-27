import os
import pickle
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
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

SymbolSeq: TypeAlias = tuple[bytes, ...]
SymbolSeqCounts: TypeAlias = Counter[SymbolSeq]
BytePair: TypeAlias = tuple[bytes, bytes]
BytePairCounts: TypeAlias = dict[BytePair, int]

NUM_PROCESSES = cpu_count() - 1
NUM_BASE_BYTES = 256  # Single-byte tokens (0x00-0xFF)


class Pretokenizer:
    def __init__(self):
        self.PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        self.BYTE_CACHE = {i: bytes([i]) for i in range(NUM_BASE_BYTES)}

    def __call__(self, doc: str) -> dict[SymbolSeq, int]:
        pre_tokenized_dict = defaultdict(int)
        for token in re.finditer(self.PAT, doc):
            key = tuple(self.BYTE_CACHE[b] for b in token[0].encode("utf-8"))
            pre_tokenized_dict[key] += 1
        return pre_tokenized_dict


class ParallelPretokenizer:
    def __init__(self):
        self.pretokenizer = Pretokenizer()

    def __call__(self, chunks: list[str]) -> SymbolSeqCounts:
        pretokenised_chunks = self._parallel_pretokenise_chunks(chunks)
        return self._merge_pretokenised_chunks(pretokenised_chunks)

    def _merge_pretokenised_chunks(
        self,
        pretokenised_chunks: Iterator[list[dict[SymbolSeq, int]]],
    ) -> SymbolSeqCounts:
        merged = Counter()
        for chunk_dicts in tqdm(pretokenised_chunks, desc="Merging pretokenized docs"):
            for doc_counts in chunk_dicts:
                merged.update(doc_counts)
        return merged

    def _parallel_pretokenise_chunks(self, chunks: list[str]) -> Iterator[list[dict[SymbolSeq, int]]]:
        with Pool(NUM_PROCESSES) as p:
            yield from tqdm(
                p.imap_unordered(self._pretokenise_chunks, chunks, chunksize=1),
                total=len(chunks),
                desc="Pretokenising chunks",
            )

    def _pretokenise_chunks(self, chunk: str) -> list[dict[SymbolSeq, int]]:
        docs = [d for d in chunk.split("<|endoftext|>") if len(d) > 0]
        return [self.pretokenizer(d) for d in docs]


class BPETrainer:
    def __init__(self):
        self.parallel_pretokenizer = ParallelPretokenizer()

    def train(
        self,
        input_path: str | os.PathLike,
        vocab_size: int,
        special_tokens: list[str],
    ) -> tuple[dict[int, bytes], list[BytePair]]:
        vocab = self._init_vocab(special_tokens)
        chunks = self._read_input_in_chunks(input_path)

        # Init state for efficient incremental merging
        self.symbol_seq_counts = self.parallel_pretokenizer(chunks)
        self.bytepair2seqs = self._bytepairs_to_symbol_seqs()
        self.seq2bytepair = self._symbol_seqs_to_bytepairs()
        self.bytepair_counts = self._to_bytepair_counts(self.symbol_seq_counts)

        merges = []
        for _ in tqdm(range(vocab_size - NUM_BASE_BYTES - len(special_tokens)), desc="Merging bytepairs"):
            merges.append(self._bp_merging())

        vocab_len = len(vocab)
        for i, merge in enumerate(merges):
            vocab[vocab_len + i] = b"".join(merge)
        return vocab, merges

    # --- Initialisation helpers ---

    def _init_vocab(self, special_tokens: list[str]) -> dict[int, bytes]:
        vocab = {i: bytes([i]) for i in range(NUM_BASE_BYTES)}
        for i, special_token in enumerate(special_tokens):
            vocab[NUM_BASE_BYTES + i] = special_token.encode("utf-8")
        return vocab

    def _read_input_in_chunks(self, input_path: str | os.PathLike) -> list[str]:
        chunks = []
        with open(input_path, "rb") as f:
            boundaries = find_chunk_boundaries(f, NUM_PROCESSES, b"<|endoftext|>")
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                chunk = f.read(end - start).decode("utf-8", errors="ignore")
                chunks.append(chunk)
        return chunks

    def _bytepairs_to_symbol_seqs(self) -> dict[BytePair, set[SymbolSeq]]:
        bytepair2seqs = defaultdict(set)
        for symbol_seq in self.symbol_seq_counts:
            for pair in pairwise(symbol_seq):
                bytepair2seqs[pair].add(symbol_seq)
        return bytepair2seqs

    def _symbol_seqs_to_bytepairs(self) -> dict[SymbolSeq, Counter[BytePair]]:
        seq2bytepair = {}
        for symbol_seq, freq in self.symbol_seq_counts.items():
            pair_counts = Counter()
            for pair in pairwise(symbol_seq):
                pair_counts[pair] += freq
            seq2bytepair[symbol_seq] = pair_counts
        return seq2bytepair

    def _to_bytepair_counts(self, symbol_seq_counts: SymbolSeqCounts) -> BytePairCounts:
        counts = defaultdict(int)
        for token, freq in symbol_seq_counts.items():
            for pair in pairwise(token):
                counts[pair] += freq
        return counts

    # --- Merging logic ---

    def _bp_merging(self) -> BytePair:
        best_pair = self._most_frequent_bytepair()
        # Convert to list once so new_seqs and seqs_to_merge share the same ordering
        seqs_to_merge = list(self.bytepair2seqs[best_pair])

        old_bytepairs_to_remove = self._find_bytepairs_to_remove(seqs_to_merge)
        new_seqs = self._merge_bytepairs_in_seqs(seqs_to_merge, best_pair)
        new_bytepair_counts = self._update_mappings_and_count_new_bytepairs(zip(new_seqs, seqs_to_merge))
        self._update_bytepair_counts(old_bytepairs_to_remove, new_bytepair_counts)
        self._update_bytepair2seqs(seqs_to_merge, old_bytepairs_to_remove)
        return best_pair

    def _most_frequent_bytepair(self) -> BytePair:
        # One pass: maximize frequency first, then lexicographic byte-pair order.
        return max(self.bytepair_counts.items(), key=lambda kv: (kv[1], kv[0]))[0]

    def _find_bytepairs_to_remove(self, seqs_to_merge: list[SymbolSeq]) -> Counter[BytePair]:
        old_bytepairs_to_remove = Counter()
        for seq in seqs_to_merge:
            old_bytepairs_to_remove += self.seq2bytepair[seq]
        return old_bytepairs_to_remove

    def _merge_bytepairs_in_seqs(self, seqs_to_merge: list[SymbolSeq], best_pair: BytePair) -> list[SymbolSeq]:
        new_seqs = []
        for seq in seqs_to_merge:
            new_seq = self._merge_in_new_symbol_seq(seq, best_pair)
            new_seqs.append(new_seq)
            freq = self.symbol_seq_counts.pop(seq)
            self.symbol_seq_counts[new_seq] += freq
        return new_seqs

    def _merge_in_new_symbol_seq(self, symbol_seq: SymbolSeq, to_merge: BytePair) -> SymbolSeq:
        out, i = [], 0
        while i < len(symbol_seq):
            if (i + 1 < len(symbol_seq)) and ((symbol_seq[i], symbol_seq[i + 1]) == to_merge):
                out.append(b"".join(to_merge))
                i += 2
            else:
                out.append(symbol_seq[i])
                i += 1
        return tuple(out)

    # --- Update helpers ---

    def _update_mappings_and_count_new_bytepairs(
        self, new2old_seqs: Iterable[tuple[SymbolSeq, SymbolSeq]]
    ) -> Counter[BytePair]:
        new_bytepair_counts = Counter()
        for new_seq, old_seq in new2old_seqs:
            seq_bytepairs = self._to_bytepair_counts({new_seq: self.symbol_seq_counts[new_seq]})
            new_bytepair_counts += Counter(seq_bytepairs)
            del self.seq2bytepair[old_seq]
            self.seq2bytepair[new_seq] = seq_bytepairs
            for bp in seq_bytepairs:
                self.bytepair2seqs[bp].add(new_seq)
        return new_bytepair_counts

    def _update_bytepair_counts(
        self, old_bytepairs_to_remove: Counter[BytePair], new_bytepair_counts: Counter[BytePair]
    ) -> None:
        for pair, freq in old_bytepairs_to_remove.items():
            self.bytepair_counts[pair] -= freq
        for pair, freq in new_bytepair_counts.items():
            self.bytepair_counts[pair] += freq

        touched = set(old_bytepairs_to_remove) | set(new_bytepair_counts)
        for pair in touched:
            if self.bytepair_counts.get(pair) == 0:
                del self.bytepair_counts[pair]

    def _update_bytepair2seqs(self, seqs_to_merge: list[SymbolSeq], old_bytepairs_to_remove: Counter[BytePair]) -> None:
        for old_bytepair in old_bytepairs_to_remove:
            seqs = self.bytepair2seqs.get(old_bytepair)
            if seqs is None:
                continue
            seqs.difference_update(seqs_to_merge)
            if not seqs:
                del self.bytepair2seqs[old_bytepair]


@app.command()
def train_bpe(
    input_path: Path,
    vocab_size: int,
    special_tokens: list[str],
    safe: bool = False,
) -> tuple[dict[int, bytes], list[BytePair]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.
        safe (bool): If True, the training process will be safe and will not overwrite existing files.
            If False, the training process will overwrite existing files.

    Returns:
        tuple[dict[int, bytes], list[BytePair]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    vocab, merges = BPETrainer().train(input_path, vocab_size, special_tokens)
    if safe:
        vocab_path = input_path.with_suffix(".pkl").with_stem("vocab_" + input_path.stem)
        merges_path = input_path.with_suffix(".pkl").with_stem("merges_" + input_path.stem)
        with open(vocab_path, "wb") as f:
            pickle.dump(vocab, f)
        with open(merges_path, "wb") as f:
            pickle.dump(merges, f)
    return vocab, merges


if __name__ == "__main__":
    app()
