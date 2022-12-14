import re
import string
from functools import lru_cache
from typing import Any, Dict, Iterable, List

from nltk import FreqDist, everygrams, pos_tag, sent_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer


def is_stop_word(s: str, stop_words: Iterable[str] = []) -> bool:
    stop_words = [*set(stopwords.words("english"))
                  .union(stop_words)
                  .union(["'s", "'ll", "n't", "'d", "'ve", "'m", "'re", "'"])]
    return s in stop_words


def extract_name(s: str) -> str:
    s = re.sub(
        r"(?P<last>([a-z]+\s)?[A-Z][a-z]+)\,\s?(?P<first>([A-Z]([a-z]+|\.?)\ ?)+)",
        r"\g<first> \g<last>", s)
    s = re.sub(
        r"(?:(?P<name>([A-Z]([a-z]+|\.?)\s?)+\b)\s?.*)",
        r"\g<name>", s)
    return s


def fix_text_wraps(s: str) -> str:
    s = re.sub(r"-\n+", r"", s)
    s = re.sub(r"\s+", r" ", s)
    return s


@lru_cache(maxsize=None)
def strip_tags(s: str) -> str:
    return re.sub(r"<[^<]+?>", r"", s)


def get_wordnet_pos(sentence: List[str]) -> Any:
    tag_dict = {
        "J": wordnet.ADJ,
        "N": wordnet.NOUN,
        "V": wordnet.VERB,
        "R": wordnet.ADV}

    for index, tag in enumerate(sentence):
        yield sentence[index], tag_dict.get(tag[1][0].upper(), wordnet.NOUN)


def clean_text(s: str, **kwargs) -> str | Iterable[str]:
    # Kwargs
    __stop_words = kwargs.get("remove_stop_words", True)
    __lowercase = kwargs.get("lowercase", True)
    __strip_tags = kwargs.get("strip_tags", True)
    __symbols = kwargs.get("remove_symbols", True)
    __links = kwargs.get("remove_links", True)
    __punctuation = kwargs.get("remove_punctuation", True)
    __numbers = kwargs.get("remove_numbers", True)
    __lemmatize = kwargs.get("lemmatize", True)
    __as_string = kwargs.get("as_string", True)

    stop_words = kwargs.get("stop_words", [])
    punctuation = r"[{0}]".format(re.sub(r"[-']", "", string.punctuation))

    # Lowercase
    s = s.lower() if __lowercase else s

    # Strip tags
    s = strip_tags(s) if __strip_tags else s

    # Symbols
    s = s.encode("cp869", errors='ignore').decode("cp869")
    s = re.sub(
        r"["
        r"\x00-\x1f"
        r"\x7f-\xa3"
        r"\xab-\xb4"
        r"\xb9-\xbc"
        r"\xbf-\xc5"
        r"\xc8-\xce"
        r"\xd9-\xdc"
        r"\xdf"
        r"\xf7-\xf9"
        r"\xfe-\xff"
        r"]",
        r' ', s
    ) if __symbols else s

    # Links
    s = re.sub(r'https?:\/\/.*[\r\n]*', '', s) if __links else s

    # Punctuation
    s = re.sub(punctuation, " ", s) if __punctuation else s

    # line breaks
    s = fix_text_wraps(s)

    # Numerics
    s = re.sub(r"\S*\d+\S*", r"", s) if __numbers else s

    # Remove extra characteres
    s = [*filter(lambda x: len(x) > 2, s.split())]

    tokens = []
    lemmatizer = WordNetLemmatizer()
    for token, pos in get_wordnet_pos(s):
        if is_stop_word(token, stop_words):
            continue

        token = lemmatizer.lemmatize(token, pos) if __lemmatize else token

        if not is_stop_word(token, stop_words):
            tokens.append(token)

    return " ".join(tokens).strip() if __as_string else tokens


def extract_ngrams(s: str, **kwargs) -> Dict:
    sort_by = kwargs.get("sort_by", "frequency")
    reverse = kwargs.get("reverse", True)
    ngrams = {}

    s = clean_text(s, as_string=False)
    freq_dist = FreqDist(everygrams(s, 1, 3))
    for ngram, frequency in freq_dist.items():
        ngram = " ".join(ngram)
        ngrams[ngram] = ngrams.get(ngram, 0) + frequency

    return dict(sorted(
        ngrams.items(), reverse=reverse,
        key=lambda item: item[0 if sort_by == "ngram" else 1]))


def split_sentences(s: str) -> List[str]:
    return sent_tokenize(s)
