import unicodedata


def normalize_devanagari(text: str) -> str:
    """Normalizes Devanagari text to standard Unicode NFC format and cleans whitespace."""
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    # Collapse multiple spaces/tabs
    return " ".join(text.strip().split())


def levenshtein_distance(seq1, seq2):
    """Computes basic Levenshtein distance between two sequences (chars or words)."""
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = [[0] * size_y for _ in range(size_x)]

    for x in range(size_x):
        matrix[x][0] = x
    for y in range(size_y):
        matrix[0][y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x - 1] == seq2[y - 1]:
                matrix[x][y] = matrix[x - 1][y - 1]
            else:
                matrix[x][y] = min(
                    matrix[x - 1][y] + 1,  # Deletion
                    matrix[x][y - 1] + 1,  # Insertion
                    matrix[x - 1][y - 1] + 1,  # Substitution
                )
    return matrix[size_x - 1][size_y - 1]


def compute_cer(predictions: list[str], references: list[str]) -> float:
    """Computes Character Error Rate across a batch."""
    total_dist = 0
    total_ref_len = 0
    for pred, ref in zip(predictions, references):
        pred_norm = normalize_devanagari(pred)
        ref_norm = normalize_devanagari(ref)
        total_dist += levenshtein_distance(list(pred_norm), list(ref_norm))
        total_ref_len += max(len(list(ref_norm)), 1)
    return total_dist / total_ref_len


def compute_wer(predictions: list[str], references: list[str]) -> float:
    """Computes Word Error Rate across a batch."""
    total_dist = 0
    total_ref_len = 0
    for pred, ref in zip(predictions, references):
        pred_words = normalize_devanagari(pred).split()
        ref_words = normalize_devanagari(ref).split()
        total_dist += levenshtein_distance(pred_words, ref_words)
        total_ref_len += max(len(ref_words), 1)
    return total_dist / total_ref_len
