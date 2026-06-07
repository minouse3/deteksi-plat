from __future__ import annotations

import re
from itertools import combinations


LETTER_TO_DIGIT = {
    "O": "0",
    "Q": "0",
    "D": "0",
    "I": "1",
    "L": "1",
    "T": "2",
    "Z": "2",
    "S": "5",
    "B": "8",
    "G": "6",
}

DIGIT_TO_LETTER = {
    "0": "O",
    "1": "I",
    "2": "Z",
    "5": "S",
    "6": "G",
    "8": "B",
}


def _as_letter(char: str) -> tuple[str, int] | None:
    if char.isalpha():
        return char, 0
    if char in DIGIT_TO_LETTER:
        return DIGIT_TO_LETTER[char], 5
    return None


def _as_digit(char: str) -> tuple[str, int] | None:
    if char.isdigit():
        return char, 0
    if char in LETTER_TO_DIGIT:
        return LETTER_TO_DIGIT[char], 2
    return None


def normalize_plate_number(raw_text: str | None) -> str | None:
    if not raw_text:
        return None

    text = re.sub(r"[^A-Z0-9]", "", raw_text.upper())
    if not text:
        return None

    prefix_match = re.match(r"^[A-Z]{1,2}", text)
    if prefix_match:
        prefix_end = prefix_match.end()
        if (
            len(text) > prefix_end + 1
            and text[prefix_end] == "0"
            and text[prefix_end + 1] in LETTER_TO_DIGIT
            and LETTER_TO_DIGIT[text[prefix_end + 1]] != "0"
        ):
            text = text[:prefix_end] + text[prefix_end + 1 :]
    text = re.sub(r"(?<=\d)0(?=[A-Z]{2,3}$)", "", text)

    best: tuple[int, str] | None = None
    n = len(text)

    # Indonesian private plates are generally: 1-2 letters, 1-4 digits, 1-3 letters.
    # Build candidates as ordered subsequences so OCR noise components can be skipped.
    for p1 in range(n):
        first = _as_letter(text[p1])
        if first is None:
            continue
        for prefix_len in [1, 2]:
            prefix_positions = [p1]
            prefix = first[0]
            cost = first[1] + (p1 * 4)

            if prefix_len == 2:
                found = None
                for p2 in range(p1 + 1, n):
                    second = _as_letter(text[p2])
                    if second is not None:
                        found = (p2, second)
                        break
                if found is None:
                    continue
                prefix_positions.append(found[0])
                prefix += found[1][0]
                cost += found[1][1] + (found[0] - p1 - 1)

            start = prefix_positions[-1] + 1
            digit_candidates = [idx for idx in range(start, n) if _as_digit(text[idx]) is not None]
            for digit_len in range(1, min(4, len(digit_candidates)) + 1):
                for digit_positions_tuple in combinations(digit_candidates, digit_len):
                    digit_positions = list(digit_positions_tuple)
                    digits = ""
                    digit_cost = 0
                    previous = start - 1
                    for pos in digit_positions:
                        digit, char_cost = _as_digit(text[pos]) or ("", 99)
                        digits += digit
                        digit_cost += char_cost + (pos - previous - 1)
                        previous = pos

                    suffix_start = digit_positions[-1] + 1
                    suffix_candidates = [idx for idx in range(suffix_start, n) if _as_letter(text[idx]) is not None]
                    for suffix_len in range(1, min(3, len(suffix_candidates)) + 1):
                        for suffix_positions_tuple in combinations(suffix_candidates, suffix_len):
                            suffix_positions = list(suffix_positions_tuple)
                            suffix = ""
                            suffix_cost = 0
                            previous = suffix_start - 1
                            for pos in suffix_positions:
                                letter, char_cost = _as_letter(text[pos]) or ("", 99)
                                suffix += letter
                                suffix_cost += char_cost + (pos - previous - 1)
                                previous = pos

                            skipped_tail = n - suffix_positions[-1] - 1
                            candidate = f"{prefix}{digits}{suffix}"
                            total_cost = cost + digit_cost + suffix_cost + skipped_tail
                            # Prefer valid-looking compact candidates over retaining noisy extras.
                            total_cost += abs(len(digits) - 4) * 4
                            if len(suffix) == 3:
                                suffix_gaps = suffix_positions[-1] - suffix_positions[0] + 1 - len(suffix_positions)
                                total_cost += 0 if suffix_gaps == 0 else 3
                                if suffix.endswith("I") and raw_text and raw_text.upper().endswith("I"):
                                    total_cost += 4
                            elif len(suffix) != 2:
                                total_cost += 1
                            if len(digits) > 1 and digits.startswith("0"):
                                total_cost += 5
                            if prefix_len == 1 and n >= 7:
                                total_cost += 4

                            if best is None or total_cost < best[0]:
                                best = (total_cost, candidate)

    return best[1] if best is not None else text
