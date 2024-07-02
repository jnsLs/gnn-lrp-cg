from typing import Sequence


def chunker(seq: Sequence, size: int) -> Sequence:
    """Helper function for working with chunks of a specified size
    for a given sequence"""
    return (seq[pos : pos + size] for pos in range(0, len(seq), size))
