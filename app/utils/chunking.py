import re
from typing import List


def smart_chunk(text: str, title: str = None, max_size: int = 500) -> List[str]:
    """
    Structure-preserving chunker.
    Keeps legal meaning intact instead of blind token splitting.
    """

    paragraphs = re.split(r"\n\s*\n", text)
    chunks = []
    current = ""

    for p in paragraphs:
        if len(current) + len(p) < max_size:
            current += "\n" + p
        else:
            chunks.append(current.strip())
            current = p

    if current:
        chunks.append(current.strip())

    if title:
        return [f"{title}\n{c}" for c in chunks]

    return chunks