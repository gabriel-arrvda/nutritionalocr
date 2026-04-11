from __future__ import annotations

from collections.abc import Iterable, Sequence


def _edit_distance(source: Sequence[str], target: Sequence[str]) -> int:
    rows = len(source) + 1
    cols = len(target) + 1
    dp = [[0] * cols for _ in range(rows)]
    for i in range(rows):
        dp[i][0] = i
    for j in range(cols):
        dp[0][j] = j

    for i in range(1, rows):
        for j in range(1, cols):
            cost = 0 if source[i - 1] == target[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[-1][-1]


def _safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator > 0 else 0.0


def compute_cer_wer_metrics(
    samples: Iterable[dict[str, str]],
) -> dict[str, dict[str, float] | dict[str, dict[str, float]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for sample in samples:
        language = sample["language"]
        grouped.setdefault(language, []).append(sample)

    per_language: dict[str, dict[str, float]] = {}
    total_char_errors = total_chars = 0
    total_word_errors = total_words = 0

    for language, language_samples in grouped.items():
        char_errors = chars = 0
        word_errors = words = 0
        for sample in language_samples:
            reference_text = sample["reference_text"]
            predicted_text = sample["predicted_text"]
            char_errors += _edit_distance(list(reference_text), list(predicted_text))
            chars += len(reference_text)
            reference_words = reference_text.split()
            predicted_words = predicted_text.split()
            word_errors += _edit_distance(reference_words, predicted_words)
            words += len(reference_words)

        per_language[language] = {
            "cer": _safe_ratio(char_errors, chars),
            "wer": _safe_ratio(word_errors, words),
        }
        total_char_errors += char_errors
        total_chars += chars
        total_word_errors += word_errors
        total_words += words

    return {
        "overall": {
            "cer": _safe_ratio(total_char_errors, total_chars),
            "wer": _safe_ratio(total_word_errors, total_words),
        },
        "per_language": per_language,
    }


def compute_detection_metrics(
    *,
    true_positives: int,
    false_positives: int,
    false_negatives: int,
) -> dict[str, float]:
    precision = _safe_ratio(true_positives, true_positives + false_positives)
    recall = _safe_ratio(true_positives, true_positives + false_negatives)
    denom = precision + recall
    f1 = (2.0 * precision * recall / denom) if denom > 0 else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
