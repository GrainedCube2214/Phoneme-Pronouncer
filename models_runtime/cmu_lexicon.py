#cmu_lexicon.py
import re
from nltk.corpus import cmudict

_cmud = cmudict.dict()

# Map ARPABET (with stress) to your lowercase TIMIT-like set
# Simplest version: strip stress digits, lowercase.
# You can refine this to match exactly your 39-phone mapping later.
def arpabet_to_timit_phone(symbol: str) -> str:
    # e.g. 'IH0' -> 'ih'
    base = re.sub(r"\d", "", symbol)  # remove stress number
    return base.lower()

def word_to_phones(word: str) -> list[str] | None:
    """
    Look up canonical phoneme sequence for a word using CMUdict.
    Returns a list of timit-style phones, or None if not found.
    """
    w = word.lower()
    if w not in _cmud:
        return None

    # Take the first pronunciation variant
    arp_seq = _cmud[w][0]  # list of symbols like ['IH1', 'N']

    phones = [arpabet_to_timit_phone(s) for s in arp_seq]
    return phones

# from cmu_lexicon import word_to_phones

# print(word_to_phones("zebra"))   # e.g. ['z', 'iy', 'b', 'r', 'ah']
