import json
import re
import threading
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

try:
    import gradio as gr
except ImportError:
    gr = None

try:
    import numpy as np
except ImportError:
    np = None

try:
    import torchaudio as ta
except Exception:
    ta = None

try:
    from faster_whisper import WhisperModel
except Exception:
    WhisperModel = None

from .config import safe_name
from .session import format_chunk_table, get_chunk, save_session, session_dir, status_message

WHISPER_MODELS: dict[str, Any] = {}
WHISPER_MODELS_LOCK = threading.Lock()


ONES = [
    "", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
    "seventeen", "eighteen", "nineteen",
]
TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]

ORDINAL_SPECIAL = {
    "one": "first",
    "two": "second",
    "three": "third",
    "five": "fifth",
    "eight": "eighth",
    "nine": "ninth",
    "twelve": "twelfth",
}

CONTRACTIONS = {
    r"\baren't\b": "are not",
    r"\bcan't\b": "can not",
    r"\bcannot\b": "can not",
    r"\bcouldn't\b": "could not",
    r"\bdidn't\b": "did not",
    r"\bdoesn't\b": "does not",
    r"\bdon't\b": "do not",
    r"\bhadn't\b": "had not",
    r"\bhasn't\b": "has not",
    r"\bhaven't\b": "have not",
    r"\bhe'd\b": "he would",
    r"\bhe'll\b": "he will",
    r"\bhe's\b": "he is",
    r"\bi'd\b": "i would",
    r"\bi'll\b": "i will",
    r"\bi'm\b": "i am",
    r"\bi've\b": "i have",
    r"\bisn't\b": "is not",
    r"\bit'd\b": "it would",
    r"\bit'll\b": "it will",
    r"\bit's\b": "it is",
    r"\blet's\b": "let us",
    r"\bmightn't\b": "might not",
    r"\bmustn't\b": "must not",
    r"\bshan't\b": "shall not",
    r"\bshe'd\b": "she would",
    r"\bshe'll\b": "she will",
    r"\bshe's\b": "she is",
    r"\bshouldn't\b": "should not",
    r"\bthat'd\b": "that would",
    r"\bthat's\b": "that is",
    r"\bthere's\b": "there is",
    r"\bthey'd\b": "they would",
    r"\bthey'll\b": "they will",
    r"\bthey're\b": "they are",
    r"\bthey've\b": "they have",
    r"\bwasn't\b": "was not",
    r"\bwe'd\b": "we would",
    r"\bwe'll\b": "we will",
    r"\bwe're\b": "we are",
    r"\bwe've\b": "we have",
    r"\bweren't\b": "were not",
    r"\bwhat'll\b": "what will",
    r"\bwhat're\b": "what are",
    r"\bwhat's\b": "what is",
    r"\bwhat've\b": "what have",
    r"\bwhere's\b": "where is",
    r"\bwho'd\b": "who would",
    r"\bwho'll\b": "who will",
    r"\bwho're\b": "who are",
    r"\bwho's\b": "who is",
    r"\bwho've\b": "who have",
    r"\bwon't\b": "will not",
    r"\bwouldn't\b": "would not",
    r"\byou'd\b": "you would",
    r"\byou'll\b": "you will",
    r"\byou're\b": "you are",
    r"\byou've\b": "you have",
    r"\bgonna\b": "going to",
    r"\bwanna\b": "want to",
    r"\bgotta\b": "got to",
    r"\bkinda\b": "kind of",
    r"\bsorta\b": "sort of",
    r"\blemme\b": "let me",
    r"\bgimme\b": "give me",
    r"\boutta\b": "out of",
    r"\bcuz\b": "because",
    r"\b'cause\b": "because",
    r"\bok\b": "okay",
}

FILLER_WORDS = {"um", "uh", "er", "ah", "hmm", "hm", "mm", "mhm", "ha", "haha", "hahaha", "heh"}

# Spoken-form abbreviations, applied symmetrically so "Dr." and "Doctor" compare equal
ABBREVIATIONS = {
    r"\bdr\b": "doctor",
    r"\bmr\b": "mister",
    r"\bmrs\b": "missus",
    r"\bprof\b": "professor",
    r"\bvs\b": "versus",
    r"\betc\b": "et cetera",
    r"\bapprox\b": "approximately",
}

ROMAN_NUMERALS = {
    "II": 2, "III": 3, "IV": 4, "VI": 6, "VII": 7, "VIII": 8, "IX": 9, "XI": 11, "XII": 12,
    "XIII": 13, "XIV": 14, "XV": 15, "XVI": 16, "XVII": 17, "XVIII": 18, "XIX": 19, "XX": 20,
}

UNIT_WORDS = {
    "kg": "kilograms", "km": "kilometers", "cm": "centimeters", "mm": "millimeters",
    "mg": "milligrams", "ml": "milliliters", "lb": "pounds", "lbs": "pounds", "ft": "feet",
    "mph": "miles per hour", "kph": "kilometers per hour", "gb": "gigabytes", "mb": "megabytes",
    "tb": "terabytes", "ghz": "gigahertz", "mhz": "megahertz", "kw": "kilowatts",
}

CURRENCY_UNITS = {
    "$": ("dollar", "dollars", "cent", "cents"),
    "£": ("pound", "pounds", "penny", "pence"),
    "€": ("euro", "euros", "cent", "cents"),
}
SCALE_ALIASES = {"k": "thousand", "m": "million", "bn": "billion"}

# Spelled-out number words; a dropped or changed one fails the chunk on its own
NUMBER_WORDS = set(ONES[1:]) | set(TENS[2:]) | {"zero", "hundred", "thousand", "million", "billion", "trillion"}

NUMBER_SLOT_RE = re.compile(r"^zslot([a-y]+)z$")

# Whisper models offered in the UI. large-v3-turbo needs faster-whisper>=1.1.
WHISPER_MODEL_CHOICES = [
    "large-v3-turbo",
    "distil-large-v3",
    "large-v3",
    "medium.en",
    "small.en",
    "base.en",
    "tiny.en",
]
DEFAULT_WHISPER_MODEL_GPU = "large-v3-turbo"
DEFAULT_WHISPER_MODEL_CPU = "small.en"

# Minimum character similarity for two differing words to count as the same spoken word
# (proper-noun spellings like "Kaelen"/"Kaylen", British/American "colour"/"color")
FUZZY_WORD_THRESHOLD = 0.8


def _small_int_to_words(n: int) -> str:
    """Convert an integer from 0 to 999 into spoken English words."""
    parts: list[str] = []
    hundreds = n // 100
    remainder = n % 100
    if hundreds:
        parts.append(f"{ONES[hundreds]} hundred")
    if remainder:
        if remainder < 20:
            parts.append(ONES[remainder])
        else:
            tens_val = remainder // 10
            ones_val = remainder % 10
            if ones_val:
                parts.append(f"{TENS[tens_val]} {ONES[ones_val]}")
            else:
                parts.append(TENS[tens_val])
    return " ".join(parts)


def int_to_words(n: int) -> str:
    """Convert an integer to spoken English words (supports negative and up to billions)."""
    if n == 0:
        return "zero"
    if n < 0:
        return "minus " + int_to_words(-n)

    scales = [
        (1_000_000_000, "billion"),
        (1_000_000, "million"),
        (1_000, "thousand"),
        (1, ""),
    ]
    curr = n
    words: list[str] = []
    for scale, name in scales:
        if curr >= scale:
            chunk_val = curr // scale
            curr %= scale
            chunk_words = _small_int_to_words(chunk_val)
            if chunk_words:
                if name:
                    words.append(f"{chunk_words} {name}")
                else:
                    words.append(chunk_words)
    return " ".join(words)


def int_to_ordinal_words(n: int) -> str:
    """Convert an integer to spoken English ordinal words (e.g. 1st -> first, 21st -> twenty first)."""
    cardinal = int_to_words(n)
    words = cardinal.split()
    last = words[-1]
    if last in ORDINAL_SPECIAL:
        words[-1] = ORDINAL_SPECIAL[last]
    elif last.endswith("y"):
        words[-1] = last[:-1] + "ieth"
    else:
        words[-1] = last + "th"
    return " ".join(words)


def _dedupe(items: list[str]) -> list[str]:
    return list(dict.fromkeys(item for item in items if item))


def _pluralize_last(words: str) -> str:
    head, _, last = words.rpartition(" ")
    last = last[:-1] + "ies" if last.endswith("y") else last + "s"
    return f"{head} {last}".strip()


def _digit_words(digits: str, zero: str) -> str:
    return " ".join(zero if d == "0" else ONES[int(d)] for d in digits)


def paired_digit_words(n: int) -> str:
    """Read a 4-digit number in pairs, as years are spoken: 1999 -> nineteen ninety nine,
    1905 -> nineteen oh five, 1900 -> nineteen hundred, 2024 -> twenty twenty four."""
    hi, lo = divmod(n, 100)
    hi_words = int_to_words(hi)
    if lo == 0:
        return f"{hi_words} hundred"
    if lo < 10:
        return f"{hi_words} oh {ONES[lo]}"
    return f"{hi_words} {int_to_words(lo)}"


def integer_spoken_variants(digits: str) -> list[str]:
    """All common spoken forms of an integer; the first is the canonical cardinal.

    e.g. 1999 -> "one thousand nine hundred ninety nine", "nineteen ninety nine";
    2500 -> "twenty five hundred"; 1204 -> "twelve oh four"; 100 -> "a hundred".
    """
    n = int(digits)
    cardinal = int_to_words(n)
    variants = [
        cardinal,
        re.sub(r"hundred (?!thousand|million|billion)", "hundred and ", cardinal),
        re.sub(r"^one (hundred|thousand|million|billion)", r"a \1", cardinal),
    ]
    if len(digits) == 4 and n >= 1000:
        variants.append(paired_digit_words(n))
    # Whisper writes a spoken clock time without its colon: "four forty-seven" -> 447, "nine oh five" -> 905
    if len(digits) == 3 and digits[0] != "0" and 0 < n % 100 < 60:
        variants.append(paired_digit_words(n))
    if digits.startswith("0") and len(digits) > 1 or len(digits) >= 5:
        variants.append(_digit_words(digits, "oh"))
        variants.append(_digit_words(digits, "zero"))
    return _dedupe(variants)


def decimal_spoken_variants(whole: str, frac: str) -> list[str]:
    """e.g. 3.5 -> "three point five", "three and a half"; 0.5 -> "zero point five", "point five"."""
    whole_words = int_to_words(int(whole))
    frac_words = _digit_words(frac, "zero")
    variants = [f"{whole_words} point {frac_words}", f"{whole_words} point {_digit_words(frac, 'oh')}"]
    if int(whole) == 0:
        variants.append(f"point {frac_words}")
    # ...or with a dot: "four forty-seven" -> 4.47
    if len(frac) == 2 and 1 <= int(whole) <= 12 and int(frac) < 60:
        variants.append(paired_digit_words(int(whole) * 100 + int(frac)))
    fraction_names = {"5": "a half", "25": "a quarter", "75": "three quarters"}
    if frac in fraction_names:
        variants.append(f"{whole_words} and {fraction_names[frac]}")
    return _dedupe(variants)


def _number_spoken_variants(whole: str, frac: str | None) -> list[str]:
    return decimal_spoken_variants(whole, frac) if frac else integer_spoken_variants(whole)


def _slot_token(index: int) -> str:
    letters = ""
    index += 1
    while index:
        index, rem = divmod(index - 1, 25)
        letters = chr(97 + rem) + letters
    return f" zslot{letters}z "


def _slot_index(token: str) -> int | None:
    m = NUMBER_SLOT_RE.match(token)
    if not m:
        return None
    index = 0
    for ch in m.group(1):
        index = index * 25 + (ord(ch) - 97 + 1)
    return index - 1


def _normalize_tokens(value: str, filter_fillers: bool) -> tuple[list[str], list[list[str]]]:
    """Normalize text into words, leaving each number as a slot token with its spoken variants."""
    if not value:
        return [], []
    slots: list[list[str]] = []

    def slot(variants: list[str]) -> str:
        slots.append(variants)
        return _slot_token(len(slots) - 1)

    # Remove Chatterbox event tags e.g. [sigh], [laugh], [whisper], and the ASR's own
    # non-speech annotations for them e.g. (laughs), *sighs*, ♪
    value = re.sub(r"\[[^\]]+\]|\([^)]*\)|\*[^*]+\*|[♪♫]", " ", value)

    # Uppercase Roman numerals e.g. World War II, Henry VIII (before lowercasing loses the signal)
    def _roman_sub(m: re.Match) -> str:
        n = ROMAN_NUMERALS[m.group(0)]
        return slot([int_to_words(n), int_to_ordinal_words(n), f"the {int_to_ordinal_words(n)}"])

    value = re.sub(r"\b(?:" + "|".join(sorted(ROMAN_NUMERALS, key=len, reverse=True)) + r")\b", _roman_sub, value)
    value = value.lower().replace("’", "'")

    # Collapse dotted acronyms e.g. U.S. -> us, A.I. -> ai (matches Whisper's "US", "AI")
    value = re.sub(r"\b(?:[a-z]\.){2,}", lambda m: m.group(0).replace(".", ""), value)

    # Symmetrically expand contractions and spoken abbreviations
    for pat, repl in CONTRACTIONS.items():
        value = re.sub(pat, repl, value)
    for pat, repl in ABBREVIATIONS.items():
        value = re.sub(pat, repl, value)

    # Remove commas between digits e.g. 1,000 -> 1000
    value = re.sub(r"(\d),(?=\d{3}\b)", r"\1", value)

    # Units attached to numbers e.g. 5kg -> 5 kilograms
    value = re.sub(
        r"(\d)\s*(" + "|".join(sorted(UNIT_WORDS, key=len, reverse=True)) + r")\b",
        lambda m: f"{m.group(1)} {UNIT_WORDS[m.group(2)]}",
        value,
    )

    # Currency e.g. $50, $50.25, £3, $1.5 million, $20k
    def _curr_sub(m: re.Match) -> str:
        unit, units, sub_unit, sub_units = CURRENCY_UNITS[m.group(1)]
        whole, frac, scale = m.group(2), m.group(3), m.group(4)
        if scale:
            scale = SCALE_ALIASES.get(scale, scale)
            variants = [f"{amount} {scale} {units}" for amount in _number_spoken_variants(whole, frac)]
            variants += [f"{amount} {scale}" for amount in _number_spoken_variants(whole, frac)]
            return slot(_dedupe(variants))
        main = unit if int(whole) == 1 else units
        variants = [f"{amount} {main}" for amount in integer_spoken_variants(whole)]
        if frac and len(frac) <= 2 and int(frac) > 0:
            cents_val = int(frac.ljust(2, "0"))
            cents_words = int_to_words(cents_val)
            sub = sub_unit if cents_val == 1 else sub_units
            whole_words = int_to_words(int(whole))
            variants = [
                f"{whole_words} {main} and {cents_words} {sub}",
                f"{whole_words} {main} {cents_words} {sub}",
                f"{whole_words} {main} {cents_words}",
                f"{whole_words} {cents_words}",
            ] + [f"{amount} {units}" for amount in decimal_spoken_variants(whole, frac)]
        # Whisper writes "$" where the speaker never says the unit: "one hundred is rent" -> "$100 is rent",
        # "three to five hundred dollars" -> "$3 to $500"; the bare amount is a correct reading too
        if scale:
            variants += [f"{amount} {scale}" for amount in _number_spoken_variants(whole, frac)]
        else:
            variants += _number_spoken_variants(whole, frac)
        return slot(_dedupe(variants))

    value = re.sub(
        r"([$£€])(\d+)(?:\.(\d+))?(?:\s*(thousand|million|billion|trillion|bn|k|m)\b)?", _curr_sub, value
    )

    # Percentages e.g. 50% -> 50 percent
    value = re.sub(r"(\d)\s*%", r"\1 percent", value)

    # Decades e.g. 1990s, '90s -> nineteen nineties, nineties
    def _decade_sub(m: re.Match) -> str:
        digits = m.group(1)
        n = int(digits)
        if len(digits) == 4:
            hi, lo = divmod(n, 100)
            hi_words = int_to_words(hi)
            first = f"{hi_words} hundreds" if lo == 0 else f"{hi_words} {_pluralize_last(int_to_words(lo))}"
            return slot(_dedupe([first, _pluralize_last(int_to_words(n))]))
        return slot([_pluralize_last(int_to_words(n))])

    value = re.sub(r"(?<![\w.])'?(\d{1,3}0)'?s\b", _decade_sub, value)

    # Times e.g. 10:30 -> ten thirty, 10:05 -> ten oh five, 10:00 -> ten o clock
    def _time_sub(m: re.Match) -> str:
        hour_words = int_to_words(int(m.group(1)))
        minute = int(m.group(2))
        if minute == 0:
            return slot([hour_words, f"{hour_words} o clock", f"{hour_words} hundred"])
        if minute < 10:
            return slot([f"{hour_words} oh {ONES[minute]}", f"{hour_words} {ONES[minute]}", f"{hour_words} zero {ONES[minute]}"])
        return slot([f"{hour_words} {int_to_words(minute)}"])

    value = re.sub(r"\b(\d{1,2}):(\d{2})\b", _time_sub, value)

    # Fractions and slash dates e.g. 1/2 -> one half, 3/4 -> three quarters, 9/11 -> nine eleven
    def _fraction_sub(m: re.Match) -> str:
        num, den = int(m.group(1)), int(m.group(2))
        num_words, den_words = int_to_words(num), int_to_words(den)
        variants = []
        if 0 < num < den <= 10:
            den_name = {2: "half", 4: "quarter"}.get(den, int_to_ordinal_words(den))
            if num > 1:
                den_name = "halves" if den == 2 else den_name + "s"
            variants += [f"{num_words} {den_name}", f"a {den_name}" if num == 1 else ""]
            if den == 4:
                variants.append(f"{num_words} {int_to_ordinal_words(4)}{'s' if num > 1 else ''}")
            if (num, den) == (1, 2):
                variants.append("half")
        variants += [f"{num_words} {den_words}", f"{num_words} over {den_words}"]
        return slot(_dedupe(variants))

    value = re.sub(r"\b(\d+)/(\d+)\b", _fraction_sub, value)

    # Ordinals e.g. 1st, 2nd, 3rd, 21st, 100th
    value = re.sub(r"\b(\d+)(?:st|nd|rd|th)\b", lambda m: int_to_ordinal_words(int(m.group(1))), value)

    # Decimals e.g. 3.5 -> three point five, three and a half
    value = re.sub(r"\b(\d+)\.(\d+)\b", lambda m: slot(decimal_spoken_variants(m.group(1), m.group(2))), value)

    # Integers e.g. 1999 -> one thousand nine hundred ninety nine, nineteen ninety nine
    value = re.sub(r"\b\d+\b", lambda m: slot(integer_spoken_variants(m.group(0))), value)

    # Common spoken symbols
    value = re.sub(r"&", " and ", value)
    value = re.sub(r"\+", " plus ", value)
    value = re.sub(r"=", " equals ", value)
    value = re.sub(r"@", " at ", value)
    value = re.sub(r"#(\w+)", r"number \1", value)

    # Clean remaining punctuation except letters, numbers, and whitespace
    value = re.sub(r"[^a-z0-9 ]+", " ", value)
    tokens = value.split()
    if filter_fillers:
        tokens = [w for w in tokens if w not in FILLER_WORDS]
    return tokens, slots


def _expand_tokens(
    tokens: list[str], slots: list[list[str]], choices: list[int]
) -> tuple[list[str], list[tuple[int, int]]]:
    """Expand slot tokens into words; also return each number's (start, end) word span."""
    words: list[str] = []
    number_spans: list[tuple[int, int]] = []
    for token in tokens:
        index = _slot_index(token)
        if index is None:
            words.extend(token.split())
            continue
        start = len(words)
        words.extend(slots[index][choices[index]].split())
        number_spans.append((start, len(words)))
    return words, number_spans


def normalize_validation_text(value: str, filter_fillers: bool = True) -> str:
    """Spotless text normalization for Whisper ASR comparison.

    Expands numbers to words, ordinals, currency, percentages, contractions,
    and removes Chatterbox event tags ([sigh], [laugh]) and common filler words (um, uh).
    Numbers take their canonical spoken form; see resolve_expected_words for variants.
    """
    tokens, slots = _normalize_tokens(value, filter_fillers)
    return " ".join(_expand_tokens(tokens, slots, [0] * len(slots))[0])


def resolve_number_words(text: str, reference_words: list[str]) -> tuple[list[str], list[tuple[int, int]]]:
    """Normalize text, reading each of its numbers the way the other side of the comparison has it.

    A number like 1999 has several correct readings ("one thousand nine hundred ninety nine",
    "nineteen ninety nine"); each is tried in turn and the best-aligned one is kept. This runs
    on the script (digits in the script) and on the transcript (Whisper writing "1999" for a
    script that spells out "nineteen ninety-nine").
    """
    tokens, slots = _normalize_tokens(text, filter_fillers=True)
    choices = [0] * len(slots)
    if reference_words:
        for index, variants in enumerate(slots):
            if len(variants) < 2:
                continue
            best_score = -1.0
            for variant_index in range(len(variants)):
                choices[index] = variant_index
                score = align_words(reference_words, _expand_tokens(tokens, slots, choices)[0])["score"]
                if score > best_score:
                    best_score, best_variant = score, variant_index
            choices[index] = best_variant
    return _expand_tokens(tokens, slots, choices)


def detect_repetition(words: list[str]) -> str | None:
    """Detect single-word or multi-word repetition loops in transcript."""
    if len(words) < 4:
        return None

    # Single-word loop: 4+ identical words consecutively
    consecutive = 1
    for i in range(1, len(words)):
        if words[i] == words[i - 1]:
            consecutive += 1
            if consecutive >= 4:
                return f"Word '{words[i]}' repeated {consecutive} times"
        else:
            consecutive = 1

    # Multi-word phrase loop: 2-6 words repeated 3+ times consecutively
    n = len(words)
    for k in range(2, 7):
        for i in range(n - 3 * k + 1):
            phrase = words[i : i + k]
            count = 1
            j = i + k
            while j + k <= n and words[j : j + k] == phrase:
                count += 1
                j += k
            if count >= 3:
                return f"Phrase '{' '.join(phrase)}' repeated {count} times"
    return None


def analyze_alignment(expected_words: list[str], transcript_words: list[str]) -> tuple[str, str] | None:
    """Analyze alignment to diagnose cutoffs at head or tail."""
    if not expected_words or not transcript_words:
        return None

    matcher = SequenceMatcher(None, expected_words, transcript_words, autojunk=False)
    blocks = matcher.get_matching_blocks()
    real_blocks = [b for b in blocks if b.size > 0]
    if not real_blocks:
        return None

    first_block = real_blocks[0]
    # Head cutoff: audio starts late, skipping initial 3+ words
    if first_block.a >= 3 and first_block.b <= 1:
        missing_head = expected_words[: first_block.a]
        return "head_cutoff", f"Audio started late; missing starting words: '{' '.join(missing_head[:8])}'"

    last_block = real_blocks[-1]
    last_a_end = last_block.a + last_block.size
    last_b_end = last_block.b + last_block.size

    # Early cutoff / truncation: audio ended early before completing expected text
    unmatched_expected_tail = len(expected_words) - last_a_end
    unmatched_transcript_tail = len(transcript_words) - last_b_end
    if (unmatched_expected_tail >= 2 or (unmatched_expected_tail >= 1 and len(expected_words) <= 3)) and unmatched_transcript_tail <= 1:
        missing_tail = expected_words[last_a_end:]
        return "early_cutoff", f"Audio cut off early; missing ending words: '{' '.join(missing_tail[:8])}'"

    return None


def classify_validation_failure(
    expected_words: list[str],
    transcript_words: list[str],
    missing: list[str],
    extra: list[str],
    score: float,
    passed: bool,
    number_errors: list[str] | None = None,
) -> tuple[str, str]:
    """Diagnose the root cause of a validation failure for clear UI feedback."""
    if passed:
        return "passed", "Validation passed cleanly."

    if not transcript_words:
        return "no_speech", "No speech detected in audio."

    repetition = detect_repetition(transcript_words)
    if repetition:
        return "repetition_loop", f"Repetition loop detected: {repetition}"

    if number_errors:
        listed = "; ".join(number_errors[:5])
        return "number_mismatch", f"Number not spoken as written: {listed}"

    alignment = analyze_alignment(expected_words, transcript_words)
    if alignment:
        return alignment

    if len(extra) >= max(3, int(len(expected_words) * 0.4)) and len(transcript_words) > len(expected_words):
        extra_sample = ", ".join(extra[:6])
        return "hallucination", f"Hallucination detected: {len(extra)} extra words ({extra_sample}...)"

    missing_sample = ", ".join(missing[:5]) if missing else "none"
    extra_sample = ", ".join(extra[:5]) if extra else "none"
    return "mismatch", f"Low match score ({score:.2f}). Missing: {missing_sample}; Extra: {extra_sample}"


def words_similar(a: str, b: str) -> bool:
    """True when two differing words are likely the same spoken word spelled differently."""
    if min(len(a), len(b)) < 3:
        return False
    return SequenceMatcher(None, a, b, autojunk=False).ratio() >= FUZZY_WORD_THRESHOLD


def align_words(expected_words: list[str], transcript_words: list[str]) -> dict[str, Any]:
    """Align expected and transcribed words, crediting spelling variants and compound splits.

    autojunk is disabled: with it on, any sequence of 200+ items has its frequent
    elements discarded as junk, which collapses the score of long chunks toward zero.
    """
    matcher = SequenceMatcher(None, expected_words, transcript_words, autojunk=False)
    matched_expected = 0
    matched_transcript = 0
    missing: list[str] = []
    missing_indices: list[int] = []
    extra: list[str] = []
    extra_indices: list[int] = []
    near_matches: list[tuple[str, str]] = []

    for tag, a0, a1, b0, b1 in matcher.get_opcodes():
        a_block = expected_words[a0:a1]
        b_block = transcript_words[b0:b1]
        if tag == "equal":
            matched_expected += len(a_block)
            matched_transcript += len(b_block)
            continue
        if tag == "replace":
            # Compound splits e.g. "every day" vs "everyday", "ice cream" vs "icecream"
            if "".join(a_block) == "".join(b_block):
                matched_expected += len(a_block)
                matched_transcript += len(b_block)
                near_matches.append((" ".join(a_block), " ".join(b_block)))
                continue
            # Greedy in-order pairing of spelling variants within the replaced span
            used: set[int] = set()
            for offset, word in enumerate(a_block):
                partner = next(
                    (j for j, cand in enumerate(b_block) if j not in used and words_similar(word, cand)),
                    None,
                )
                if partner is None:
                    missing.append(word)
                    missing_indices.append(a0 + offset)
                else:
                    used.add(partner)
                    matched_expected += 1
                    matched_transcript += 1
                    near_matches.append((word, b_block[partner]))
            for j, cand in enumerate(b_block):
                if j not in used:
                    extra.append(cand)
                    extra_indices.append(b0 + j)
            continue
        missing.extend(a_block)
        missing_indices.extend(range(a0, a1))
        extra.extend(b_block)
        extra_indices.extend(range(b0, b1))

    total = len(expected_words) + len(transcript_words)
    score = (matched_expected + matched_transcript) / total if total else 0.0
    return {
        "score": score,
        "missing": missing,
        "missing_indices": missing_indices,
        "extra": extra,
        "extra_indices": extra_indices,
        "near_matches": near_matches,
    }


def _number_runs(words: list[str], indices: list[int], spans: list[tuple[int, int]]) -> list[str]:
    """Group unmatched word positions that belong to a number (digit-derived span or number word)."""
    in_span = {i for start, end in spans for i in range(start, end)}
    flagged = sorted(i for i in indices if i in in_span or words[i] in NUMBER_WORDS)
    runs: list[list[int]] = []
    for i in flagged:
        if runs and i == runs[-1][-1] + 1:
            runs[-1].append(i)
        else:
            runs.append([i])
    return [" ".join(words[i] for i in run) for run in runs]


def validation_comparison(expected: str, transcript: str, threshold: float = 0.8) -> dict[str, Any]:
    # Numbers can be written either way on either side (script "nineteen ninety-nine", Whisper
    # "1999", or the reverse), so each side's numbers are read the way the other side has them
    expected_words, expected_spans = resolve_number_words(
        expected, normalize_validation_text(transcript, filter_fillers=True).split()
    )
    transcript_words, transcript_spans = resolve_number_words(transcript, expected_words)
    expected_normalized = " ".join(expected_words)
    transcript_normalized = " ".join(transcript_words)

    alignment = align_words(expected_words, transcript_words)
    if not transcript_normalized or not expected_normalized:
        word_score = 0.0
        char_score = 0.0
    else:
        word_score = alignment["score"]
        # Space-insensitive character similarity catches multi-word phonetic variants
        # e.g. "kaelen voss" vs "kay lynn vos"
        char_score = SequenceMatcher(
            None, expected_normalized.replace(" ", ""), transcript_normalized.replace(" ", ""), autojunk=False
        ).ratio()

    # Optimal composite score: word accuracy with fallback to character similarity for phonetic variants
    score = max(round(word_score, 4), round(char_score, 4))

    missing = alignment["missing"]
    extra = alignment["extra"]
    # A misread number changes meaning, so it fails the chunk even when the overall score is high
    number_errors = [
        f"expected '{run}'" for run in _number_runs(expected_words, alignment["missing_indices"], expected_spans)
    ] + [f"heard '{run}'" for run in _number_runs(transcript_words, alignment["extra_indices"], transcript_spans)]
    passed = bool(transcript_normalized) and score >= float(threshold) and not number_errors

    diagnostic_category, diagnostic_message = classify_validation_failure(
        expected_words, transcript_words, missing, extra, score, passed, number_errors
    )

    return {
        "expected_normalized": expected_normalized,
        "transcript_normalized": transcript_normalized,
        "score": score,
        "word_score": round(word_score, 4),
        "char_score": round(char_score, 4),
        "threshold": float(threshold),
        "passed": passed,
        "diagnostic_category": diagnostic_category,
        "diagnostic_message": diagnostic_message,
        "missing_words": missing[:30],
        "extra_words": extra[:30],
        "near_matches": [f"{a} ~ {b}" for a, b in alignment["near_matches"][:30]],
        "number_errors": number_errors,
        "expected_word_count": len(expected_words),
        "transcript_word_count": len(transcript_words),
    }


def get_whisper_model(model_name: str, device: str):
    if WhisperModel is None:
        raise gr.Error(
            "faster-whisper is not installed. In Kaggle, run `pip install 'faster-whisper>=1.1.0'` and restart the session."
        )

    selected_device = str(device or "cpu")
    cache_key = f"{model_name}:{selected_device}"
    with WHISPER_MODELS_LOCK:
        if cache_key not in WHISPER_MODELS:
            if selected_device.startswith("cuda"):
                device_index = int(selected_device.split(":", 1)[1]) if ":" in selected_device else 0
                WHISPER_MODELS[cache_key] = WhisperModel(
                    model_name,
                    device="cuda",
                    device_index=device_index,
                    compute_type="float16",
                )
            else:
                WHISPER_MODELS[cache_key] = WhisperModel(
                    model_name,
                    device="cpu",
                    compute_type="int8",
                )
    return WHISPER_MODELS[cache_key]


def transcribe_audio(
    audio_path: str,
    whisper_model_name: str,
    whisper_device: str,
    prompt_text: str = "",
    language: str = "en",
    vad_filter: bool = False,
) -> str:
    if not audio_path or not Path(audio_path).exists():
        if gr is not None:
            raise gr.Error("Generate the chunk audio before validating it.")
        raise ValueError("Generate the chunk audio before validating it.")

    model = get_whisper_model(whisper_model_name, whisper_device)

    # Prepare audio input with edge silence padding to prevent boundary truncation
    audio_input: Any = audio_path
    if ta is not None and np is not None:
        try:
            wav, sr = ta.load(audio_path)
            if wav.ndim == 2:
                wav = wav.mean(dim=0, keepdim=True)
            if sr != 16000:
                resampler = ta.transforms.Resample(orig_freq=sr, new_freq=16000)
                wav = resampler(wav)
                sr = 16000
            samples = wav.squeeze(0).float().numpy()
            # Pad 250ms silence on both sides to prevent mel spectrogram edge cutoff
            pad_samples = int(sr * 0.25)
            samples = np.pad(samples, (pad_samples, pad_samples), mode="constant")
            audio_input = samples
        except Exception:
            audio_input = audio_path

    kwargs: dict[str, Any] = {
        "beam_size": 5,
        "vad_filter": vad_filter,
        "condition_on_previous_text": False,
        # Timestamp tokens let Whisper seek to the last complete segment of each 30s window;
        # without them, chunks longer than 30s lose or duplicate words at window boundaries.
        "without_timestamps": False,
        "compression_ratio_threshold": 2.6,
        # Whisper default. Lower values drop quiet/breathy segments as "no speech",
        # which surfaces as false early_cutoff / missing-word failures.
        "no_speech_threshold": 0.6,
    }
    if vad_filter:
        kwargs["vad_parameters"] = {
            "threshold": 0.20,
            "speech_pad_ms": 400,
            "min_silence_duration_ms": 1000,
        }
    if language:
        kwargs["language"] = language

    segments, _info = model.transcribe(audio_input, **kwargs)
    return " ".join(segment.text.strip() for segment in segments).strip()


def validation_path(session: dict[str, Any], chunk: dict[str, Any]) -> Path:
    return session_dir(session) / "validation" / f"{chunk['index']:04d}.json"


def write_validation_report(session: dict[str, Any], chunk: dict[str, Any], section: str, payload: dict[str, Any]):
    path = validation_path(session, chunk)
    path.parent.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {}
    if path.exists():
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            report = {}
    report.setdefault("chunk_index", chunk["index"])
    report[section] = payload
    temp_path = path.with_suffix(".tmp.json")
    temp_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    temp_path.replace(path)
    chunk["validation_path"] = str(path)


def validate_chunk(
    session: dict[str, Any],
    chunk: dict[str, Any],
    whisper_model_name: str,
    whisper_device: str,
    validation_threshold: float,
    vad_filter: bool = False,
) -> dict[str, Any]:
    transcript = transcribe_audio(
        chunk.get("audio_path"),
        whisper_model_name,
        whisper_device,
        prompt_text=chunk.get("text", ""),
        language="en",
        vad_filter=vad_filter,
    )
    comparison = validation_comparison(chunk["text"], transcript, float(validation_threshold))
    result = {
        "chunk_index": chunk["index"],
        "audio_path": chunk.get("audio_path"),
        "model_name": chunk.get("model_name"),
        "whisper_model": whisper_model_name,
        "whisper_device": whisper_device,
        "transcript": transcript,
        **comparison,
    }
    write_validation_report(session, chunk, "whisper", result)

    chunk["transcript"] = transcript
    chunk["text_score"] = comparison["score"]
    chunk["validation_status"] = "passed" if comparison["passed"] else "needs_review"
    chunk["validation_category"] = comparison.get("diagnostic_category")
    chunk["validation_error"] = None if comparison["passed"] else comparison.get("diagnostic_message")
    chunk["status"] = "validated" if comparison["passed"] else "needs_review"
    return result


def validation_details(chunk: dict[str, Any] | None) -> str:
    if not chunk:
        return "No chunk selected."
    sections = []
    if chunk.get("validation_status"):
        category = chunk.get("validation_category")
        category_map = {
            "passed": "✅ Passed",
            "no_speech": "⚠️ No Speech",
            "repetition_loop": "⚠️ Repetition Loop",
            "early_cutoff": "⚠️ Early Cutoff",
            "number_mismatch": "⚠️ Number Mismatch",
            "head_cutoff": "⚠️ Head Cutoff",
            "hallucination": "⚠️ Hallucination",
            "mismatch": "⚠️ Word Mismatch",
        }
        cat_str = f" ({category_map.get(category, category)})" if category else ""
        icon = "✅" if chunk["validation_status"] == "passed" else "⚠️"
        sections.append(
            f"**Whisper:** {icon} `{chunk['validation_status']}`{cat_str}  \n"
            f"**Text score:** `{chunk.get('text_score', 0):.3f}`  \n"
            f"**Transcript:** \"{chunk.get('transcript') or '(empty)'}\"  \n"
            f"**Details:** {chunk.get('validation_error') or 'Validation passed cleanly.'}"
        )
    if chunk.get("audio_quality_status"):
        aq_icon = "✅" if chunk["audio_quality_status"] == "passed" else "⚠️"
        sections.append(
            f"**Audio quality:** {aq_icon} `{chunk['audio_quality_status']}`  \n"
            f"**Details:** {chunk.get('audio_quality_error') or 'No audio quality issues detected.'}"
        )
    return "\n\n".join(sections) or "This chunk has not been validated yet."


def check_audio_quality(
    session: dict[str, Any],
    chunk: dict[str, Any],
    silence_threshold_db: float,
    max_silence_ms: float,
    max_clip_fraction: float,
    min_duration_s: float,
    max_duration_s: float,
    min_rms_dbfs: float,
) -> dict[str, Any]:
    if ta is None:
        raise gr.Error("torchaudio is required for audio quality checks.")
    audio_path = chunk.get("audio_path")
    if not audio_path or not Path(audio_path).exists():
        raise gr.Error("Generate the chunk audio before checking its quality.")

    wav, sample_rate = ta.load(audio_path)
    samples = wav.float().numpy()
    mono = samples.mean(axis=0) if samples.ndim == 2 else samples
    duration_s = float(len(mono) / sample_rate) if sample_rate else 0.0
    peak = float(np.max(np.abs(mono))) if len(mono) else 0.0
    rms = float(np.sqrt(np.mean(np.square(mono)))) if len(mono) else 0.0
    peak_dbfs = 20.0 * np.log10(max(peak, 1e-8))
    rms_dbfs = 20.0 * np.log10(max(rms, 1e-8))
    silence_amplitude = 10.0 ** (float(silence_threshold_db) / 20.0)
    audible = np.abs(mono) > silence_amplitude
    if audible.any():
        first_audible = int(np.argmax(audible))
        last_audible = int(len(audible) - 1 - np.argmax(audible[::-1]))
        leading_silence_ms = first_audible / sample_rate * 1000.0
        trailing_silence_ms = (len(audible) - 1 - last_audible) / sample_rate * 1000.0
    else:
        leading_silence_ms = duration_s * 1000.0
        trailing_silence_ms = duration_s * 1000.0

    clip_fraction = float(np.mean(np.abs(mono) >= 0.999)) if len(mono) else 1.0
    issues: list[str] = []
    if duration_s < float(min_duration_s):
        issues.append(f"duration too short ({duration_s:.2f}s)")
    if duration_s > float(max_duration_s):
        issues.append(f"duration too long ({duration_s:.2f}s)")
    if peak < 1e-5 or not audible.any():
        issues.append("near-silent audio")
    if rms_dbfs < float(min_rms_dbfs) and audible.any():
        issues.append(f"low loudness RMS {rms_dbfs:.1f}dBFS")
    if leading_silence_ms > float(max_silence_ms):
        issues.append(f"leading silence {leading_silence_ms:.0f}ms")
    if trailing_silence_ms > float(max_silence_ms):
        issues.append(f"trailing silence {trailing_silence_ms:.0f}ms")
    if clip_fraction > float(max_clip_fraction):
        issues.append(f"clipping fraction {clip_fraction:.5f}")

    score = max(0.0, 1.0 - min(1.0, len(issues) / 5.0))
    result = {
        "sample_rate": int(sample_rate),
        "duration_s": round(duration_s, 4),
        "peak_dbfs": round(peak_dbfs, 3),
        "rms_dbfs": round(rms_dbfs, 3),
        "leading_silence_ms": round(leading_silence_ms, 2),
        "trailing_silence_ms": round(trailing_silence_ms, 2),
        "clip_fraction": round(clip_fraction, 6),
        "silence_threshold_db": float(silence_threshold_db),
        "max_silence_ms": float(max_silence_ms),
        "min_rms_dbfs": float(min_rms_dbfs),
        "passed": not issues,
        "issues": issues,
    }
    chunk["audio_quality_status"] = "passed" if result["passed"] else "needs_review"
    chunk["audio_quality_score"] = round(score, 4)
    chunk["audio_quality_error"] = "; ".join(issues) if issues else None
    write_validation_report(session, chunk, "audio_quality", result)
    return result


def audio_quality_details(chunk: dict[str, Any] | None) -> str:
    if not chunk or not chunk.get("audio_quality_status"):
        return "Audio quality has not been checked yet."
    return (
        f"**Audio quality:** `{chunk['audio_quality_status']}`  \n"
        f"**Details:** {chunk.get('audio_quality_error') or 'No audio quality issues detected.'}"
    )


def check_audio_selected(
    session: dict[str, Any],
    chunk_number: int,
    silence_threshold_db: float,
    max_silence_ms: float,
    max_clip_fraction: float,
    min_duration_s: float,
    max_duration_s: float,
    min_rms_dbfs: float,
    progress: Any = None,
):
    if not session:
        raise gr.Error("Create or load a session first.")
    prog = progress if callable(progress) else (lambda *args, **kwargs: None)
    prog(0.2, desc=f"Analyzing audio quality for chunk {chunk_number}...")
    chunk = get_chunk(session, int(chunk_number or 1))
    result = check_audio_quality(
        session, chunk, silence_threshold_db, max_silence_ms, max_clip_fraction, min_duration_s, max_duration_s, min_rms_dbfs
    )
    save_session(session)
    prog(1.0, desc=f"Chunk {chunk['index']} audio check: {'passed' if result['passed'] else 'needs review'}.")
    return (
        session,
        format_chunk_table(session),
        audio_quality_details(chunk),
        status_message(session, f"Checked audio for chunk {chunk['index']}: {'passed' if result['passed'] else 'needs review'}."),
    )


def check_audio_all(
    session: dict[str, Any],
    silence_threshold_db: float,
    max_silence_ms: float,
    max_clip_fraction: float,
    min_duration_s: float,
    max_duration_s: float,
    min_rms_dbfs: float,
    progress: Any = None,
):
    if not session:
        if gr is not None:
            raise gr.Error("Create or load a session first.")
        raise ValueError("Create or load a session first.")
    prog = progress if callable(progress) else (lambda *args, **kwargs: None)
    chunks = [chunk for chunk in session.get("chunks", []) if chunk.get("audio_path") and chunk["status"] != "excluded"]
    if not chunks:
        if gr is not None:
            raise gr.Error("Generate at least one chunk before checking audio quality.")
        raise ValueError("Generate at least one chunk before checking audio quality.")
    passed = 0
    for chunk_index, chunk in enumerate(chunks, start=1):
        try:
            result = check_audio_quality(
                session, chunk, silence_threshold_db, max_silence_ms, max_clip_fraction, min_duration_s, max_duration_s, min_rms_dbfs
            )
            passed += int(result["passed"])
        except Exception as exc:
            chunk["audio_quality_status"] = "error"
            chunk["audio_quality_error"] = str(exc)
        save_session(session)
        prog(chunk_index / len(chunks), desc=f"Checked audio {chunk_index}/{len(chunks)}")
    return (
        session,
        format_chunk_table(session),
        status_message(session, f"Checked audio for {len(chunks)} chunk(s): {passed} passed, {len(chunks) - passed} need review."),
    )


def validate_selected_chunk(
    session: dict[str, Any],
    chunk_number: int,
    whisper_model_name: str,
    whisper_device: str,
    validation_threshold: float,
    enabled: bool,
    progress: Any = None,
):
    if not session:
        raise gr.Error("Create or load a session first.")
    if not enabled:
        raise gr.Error("Enable Whisper validation first, or continue without validation.")
    prog = progress if callable(progress) else (lambda *args, **kwargs: None)
    prog(0.2, desc=f"Validating chunk {chunk_number} with {whisper_model_name}...")
    chunk = get_chunk(session, int(chunk_number or 1))
    result = validate_chunk(session, chunk, whisper_model_name, whisper_device, float(validation_threshold))
    save_session(session)
    prog(1.0, desc=f"Chunk {chunk['index']} validated: {chunk['validation_status']}.")
    diag_cat = result.get("diagnostic_category")
    diag_suffix = f" ({diag_cat})" if diag_cat and diag_cat != "passed" else ""
    return (
        session,
        format_chunk_table(session),
        chunk.get("transcript") or "",
        validation_details(chunk),
        status_message(session, f"Validated chunk {chunk['index']}: {chunk['validation_status']}{diag_suffix} (score: {result['score']:.3f})."),
    )


def validate_all_chunks(
    session: dict[str, Any],
    whisper_model_name: str,
    whisper_device: str,
    validation_threshold: float,
    enabled: bool,
    progress: Any = None,
):
    if not session:
        if gr is not None:
            raise gr.Error("Create or load a session first.")
        raise ValueError("Create or load a session first.")
    if not enabled:
        if gr is not None:
            raise gr.Error("Enable Whisper validation first, or continue without validation.")
        raise ValueError("Enable Whisper validation first, or continue without validation.")
    prog = progress if callable(progress) else (lambda *args, **kwargs: None)
    chunks = [chunk for chunk in session.get("chunks", []) if chunk.get("audio_path") and chunk["status"] != "excluded"]
    if not chunks:
        if gr is not None:
            raise gr.Error("Generate at least one chunk before validating.")
        raise ValueError("Generate at least one chunk before validating.")

    passed = 0
    categories: dict[str, int] = {}
    first_error = None
    for chunk_index, chunk in enumerate(chunks, start=1):
        try:
            result = validate_chunk(session, chunk, whisper_model_name, whisper_device, float(validation_threshold))
            passed += int(result["passed"])
            cat = result.get("diagnostic_category", "unknown")
            categories[cat] = categories.get(cat, 0) + 1
            save_session(session)
            prog(chunk_index / len(chunks), desc=f"Validated chunk {chunk_index}/{len(chunks)}")
        except Exception as exc:
            first_error = exc
            chunk["validation_status"] = "error"
            chunk["validation_error"] = str(exc)
            save_session(session)

    issue_items = [f"{k}: {v}" for k, v in categories.items() if k != "passed"]
    issues_str = f" ({', '.join(issue_items)})" if issue_items else ""
    message = f"Validated {len(chunks)} chunk(s): {passed} passed, {len(chunks) - passed} need review{issues_str}."
    if first_error:
        message += f" First error: {first_error}"
    return session, format_chunk_table(session), status_message(session, message)


def export_validation_report(session: dict[str, Any]):
    if not session:
        raise gr.Error("Create or load a session first.")
    report_dir = session_dir(session) / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    chunks = session.get("chunks", [])
    report = {
        "session_id": session.get("session_id"),
        "project_name": session.get("project_name"),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "summary": {
            "total_chunks": len(chunks),
            "generated": sum(bool(chunk.get("audio_path")) for chunk in chunks),
            "whisper_passed": sum(chunk.get("validation_status") == "passed" for chunk in chunks),
            "whisper_needs_review": sum(chunk.get("validation_status") == "needs_review" for chunk in chunks),
            "audio_passed": sum(chunk.get("audio_quality_status") == "passed" for chunk in chunks),
            "audio_needs_review": sum(chunk.get("audio_quality_status") == "needs_review" for chunk in chunks),
        },
        "chunks": chunks,
    }
    report_path = report_dir / f"{safe_name(session.get('project_name'), 'narration')}_validation_report.json"
    temp_path = report_path.with_suffix(".tmp.json")
    temp_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    temp_path.replace(report_path)
    return str(report_path), status_message(session, f"Exported validation report: `{report_path.name}`.")


def build_validation_overview(session: dict[str, Any] | None) -> tuple[str, list[tuple[str, int]], str]:
    """Build a comprehensive validation dashboard overview, failed choices, and button label."""
    if not session or not session.get("chunks"):
        return "### 📊 Validation Health Scorecard\n*(No session active)*", [], "🔄 Regenerate Problem Chunks"

    chunks = session.get("chunks", [])
    total = len(chunks)
    generated = sum(bool(c.get("audio_path")) for c in chunks)
    excluded = sum(c.get("status") == "excluded" for c in chunks)

    whisper_validated = [c for c in chunks if c.get("validation_status") and c.get("status") != "excluded"]
    whisper_passed = [c for c in whisper_validated if c.get("validation_status") == "passed"]
    whisper_needs_review = [c for c in whisper_validated if c.get("validation_status") != "passed"]

    audio_checked = [c for c in chunks if c.get("audio_quality_status") and c.get("status") != "excluded"]
    audio_passed = [c for c in audio_checked if c.get("audio_quality_status") == "passed"]
    audio_needs_review = [c for c in audio_checked if c.get("audio_quality_status") != "passed"]

    synth_failed = [c for c in chunks if c.get("status") == "failed" and c.get("status") != "excluded"]

    w_count = len(whisper_validated)
    w_pct = f"{(len(whisper_passed) / w_count * 100):.1f}%" if w_count else "—"
    a_count = len(audio_checked)
    a_pct = f"{(len(audio_passed) / a_count * 100):.1f}%" if a_count else "—"

    total_problems = len(whisper_needs_review) + len(audio_needs_review) + len(synth_failed)

    # Headline status badge
    if not whisper_validated and not audio_checked and not synth_failed:
        badge = "⏳ *(Validation not run yet)*"
    elif total_problems == 0:
        badge = f"🎉 **100% Clean! All {len(whisper_passed)} validated chunk(s) passed**"
    else:
        badge = f"⚠️ **{total_problems} chunk(s) need attention**"

    lines = [
        f"### 📊 Validation Health Scorecard — {badge}",
        "",
        "| Metric | Total Chunks | Audio Generated | Whisper ASR Passed | Whisper Needs Review | Audio Acoustic Checks |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
        f"| **Count / Ratio** | **{total}** | **{generated}/{total}** | **{len(whisper_passed)}/{w_count or total}** ({w_pct}) | **{len(whisper_needs_review)}** | **{len(audio_passed)}/{a_count or total}** ({a_pct}) |",
        "",
    ]
    if excluded > 0:
        lines.append(f"*(Note: {excluded} chunk(s) currently marked as excluded)*\n")

    problem_choices: list[tuple[str, int]] = []
    seen_indices = set()

    if synth_failed:
        lines.append("#### 💥 Generation / Synthesis Errors:")
        for c in synth_failed:
            seen_indices.add(c["index"])
            cid = c.get("id") or f"c{c['index']:03d}"
            err = c.get("error") or "Generation error occurred"
            lines.append(f"- 💥 **Chunk {c['index']}** (`{cid}`) — *{err}*")
            problem_choices.append((f"Chunk {c['index']} ({cid}) — Generation Error", c["index"]))
        lines.append("")

    if whisper_needs_review:
        cat_counts: dict[str, int] = {}
        for c in whisper_needs_review:
            cat = c.get("validation_category") or "other"
            cat_counts[cat] = cat_counts.get(cat, 0) + 1
        cat_labels = [f"**{k.replace('_', ' ').title()}**: {v}" for k, v in sorted(cat_counts.items())]
        lines.append(f"**Diagnostic Issues Breakdown:** {', '.join(cat_labels)}")
        lines.append("")
        lines.append("#### ⚠️ Whisper Speech Verification Issues:")
        for c in whisper_needs_review:
            seen_indices.add(c["index"])
            cid = c.get("id") or f"c{c['index']:03d}"
            cat = (c.get("validation_category") or "needs_review").replace("_", " ").title()
            score = f"{c.get('text_score', 0):.2f}"
            detail = c.get("validation_error") or "Speech text mismatch"
            transcript = f' Transcribed: "{c.get("transcript")}"' if c.get("transcript") else ""
            lines.append(f"- 🔴 **Chunk {c['index']}** (`{cid}`) [Score: `{score}`] — **{cat}**: *{detail}*{transcript}")
            problem_choices.append((f"Chunk {c['index']} ({cid}) — {cat} (Score: {score})", c["index"]))
        lines.append("")

    if audio_needs_review:
        lines.append("#### 🟡 Audio Quality Issues:")
        for c in audio_needs_review:
            cid = c.get("id") or f"c{c['index']:03d}"
            detail = c.get("audio_quality_error") or "Acoustic issue"
            lines.append(f"- 🟡 **Chunk {c['index']}** (`{cid}`) — *{detail}*")
            if c["index"] not in seen_indices:
                seen_indices.add(c["index"])
                problem_choices.append((f"Chunk {c['index']} ({cid}) — Audio Quality Issue", c["index"]))
        lines.append("")

    if whisper_validated and not whisper_needs_review and not audio_needs_review and not synth_failed:
        lines.append("✅ *All validated chunks meet or exceed your text accuracy and acoustic quality criteria.*")

    btn_text = f"🔄 Regenerate {len(problem_choices)} Problem Chunk(s)" if problem_choices else "🔄 Regenerate Chunks Needing Review"
    return "\n".join(lines), problem_choices, btn_text


def build_full_report_markdown(session: dict[str, Any] | None) -> str:
    """Generate a full markdown table view of all chunks and their validation details."""
    if not session or not session.get("chunks"):
        return "*(No session active)*"
    chunks = session.get("chunks", [])
    lines = [
        f"### 📋 Detailed Verification Manifest — `{session.get('project_name', 'Narration')}`",
        "",
        "| # | ID | Status | Whisper | Score | Diagnostic | Transcript | Text Preview |",
        "| :---: | :---: | :---: | :---: | :---: | :--- | :--- | :--- |",
    ]
    for c in chunks:
        idx = c["index"]
        cid = c.get("id") or f"c{idx:03d}"
        status = c.get("status") or "pending"
        v_status = c.get("validation_status") or "unvalidated"
        icon = "✅" if v_status == "passed" else ("⚠️" if v_status == "needs_review" else ("💥" if status == "failed" else "⏳"))
        score = f"{c.get('text_score', 0):.2f}" if c.get("text_score") is not None else "—"
        cat = (c.get("validation_category") or "—").replace("_", " ").title()
        transcript = (c.get("transcript") or "—").replace("\n", " ")
        if len(transcript) > 40:
            transcript = transcript[:37] + "..."
        preview = c["text"].replace("\n", " ")
        if len(preview) > 40:
            preview = preview[:37] + "..."
        lines.append(f"| {idx} | `{cid}` | `{status}` | {icon} `{v_status}` | `{score}` | {cat} | *{transcript}* | {preview} |")
    return "\n".join(lines)
