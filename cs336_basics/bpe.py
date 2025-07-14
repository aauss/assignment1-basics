import os
import pickle
from collections import Counter, defaultdict
from itertools import pairwise
from multiprocessing import Pool
from pathlib import Path

import regex as re
import typer
from tqdm import tqdm

app = typer.Typer()


def pretokenize(doc: str) -> defaultdict[tuple[bytes], int]:
    PAT = """'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    pre_tokenised = re.finditer(PAT, doc)
    pre_tokenized_dict = defaultdict(int)
    for token in pre_tokenised:
        key = string2tuple_of_bytes(token[0])
        pre_tokenized_dict[key] += 1
    return pre_tokenized_dict


def string2tuple_of_bytes(s: str) -> tuple[bytes]:
    return tuple(bytes([i]) for i in s.encode("utf-8"))


def merge_bytes(pre_tokenized: defaultdict[tuple[bytes], int]) -> defaultdict[tuple[bytes], int]:
    conseq_token_counts = token_count_to_bpe(pre_tokenized)
    # Find lexicographically largest pair of bytes if frequencies tie
    to_merge = most_frequent_conseq_token(conseq_token_counts)

    # Update pre-tokenized dictionary
    for token in list(pre_tokenized):
        # Iterate over pairwise token
        if to_merge in list(pairwise(token)):
            new_key = merge_in_new_token(token, to_merge)
            # new_key = b",".join(token).replace(b",".join(to_merge), b"".join(to_merge)).split(b",")
            freq = pre_tokenized.pop(token)
            pre_tokenized[tuple(new_key)] = freq
    return pre_tokenized, to_merge


def merge_in_new_token(token: tuple[bytes], to_merge: tuple[bytes]):
    out, i = [], 0
    while i < len(token):
        if (i + 1 < len(token)) and ((token[i], token[i + 1]) == to_merge):
            out.append(b"".join(to_merge))
            i += 2
        else:
            out.append(token[i])
            i += 1
    return tuple(out)


def token_count_to_bpe(pre_tokenized_dict: defaultdict[str, int]) -> defaultdict[tuple[str, str], int]:
    conseq_token_counts = defaultdict(int)
    for token, freq in pre_tokenized_dict.items():
        for pair in pairwise(token):
            conseq_token_counts[pair] += freq
    return conseq_token_counts


def most_frequent_conseq_token(conseq_token_counts: defaultdict[tuple[bytes], int]) -> tuple[bytes]:
    max_token_counts = max(conseq_token_counts.values())
    most_frequent_conseq_token = (k for k, v in conseq_token_counts.items() if v == max_token_counts)
    return max(most_frequent_conseq_token)


def consecutive_merging(pre_tokenized: defaultdict[tuple[bytes], int], num_merges: int) -> list[tuple[bytes]]:
    list_merged_bytes = []
    for _ in tqdm(range(num_merges), desc="Merging bytes"):
        pre_tokenized, merged_bytes = merge_bytes(pre_tokenized)
        list_merged_bytes.append(merged_bytes)
    return list_merged_bytes


def _run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
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
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    vocab = {i: bytes([i]) for i in range(256)}
    for i, special_token in enumerate(special_tokens):
        vocab[256 + i] = special_token.encode("utf-8")
    with open(input_path) as f:
        txt = f.read()
    # Pretokenize the corpus and don't BPE special tokens
    docs = txt.split("<|endoftext|>")

    with Pool(6) as p:
        pretokenised_docs = list(tqdm(p.imap(pretokenize, docs), total=len(docs), desc="Pretokenizing"))
    pretokenized_merged = Counter()
    for d in tqdm(pretokenised_docs, desc="Merging pretokenized docs"):
        pretokenized_merged.update(d)

    new_tokens = consecutive_merging(pretokenized_merged, vocab_size - 256 - len(special_tokens))
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
