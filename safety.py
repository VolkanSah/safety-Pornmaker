# source: https://github.com/VolkanSah/safety.py see always for updates! 
# Please! will set up an ai for complex stuff and soon as vector! still crafting sorry

"""Safety layer for an adult image generator.

Policy: adult sexual content is allowed. ANYTHING that suggests a minor is blocked, strictly.
Also blocked: gore/violence. This is ONE layer: text filter + forced prompt hardening + image age check.
It is not a perfect shield (cannot judge real people, clever workarounds, new slang). Keep it updated.
"""
import re
import unicodedata

MAX_PROMPT_LEN = 1500  # long prompts are supported (Compel), but everything is scanned

_LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"})

# Whole words that suggest minors (English + German). Deliberately strict: a false alarm is better than a miss.
MINOR_WORDS = {
    "child", "children", "childlike", "kid", "kids", "kind", "kinder", "minor", "minors", "underage",
    "teen", "teens", "teenage", "teenager", "boy", "boys", "girl", "girls", "baby", "toddler", "infant",
    "loli", "lolita", "lolicon", "shota", "shotacon", "schoolgirl", "schoolboy", "school", "student",
    "classroom", "young", "youth", "juvenile", "preteen", "tween", "jailbait", "pedo", "pedophile",
    "cub", "petite", "innocent", "childish", "flatchested", "pubescent", "prepubescent", "adolescent",
    "junge", "jungen", "madchen", "jugendlich", "minderjahrig", "schuler", "schulerin", "schule",
    "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
}
# Substrings checked on the text with spaces/punctuation removed ("l o l i", "s.c.h.o.o.l.g.i.r.l")
# Careful: do not add short fragments like "teen" (would hit "nineteen").
MINOR_SUBSTRINGS = ("loli", "shota", "underage", "preteen", "schoolgirl", "schoolboy", "jailbait",
                    "pedo", "childlike", "prepubescent", "minderjahrig",
                    # childlike BODY descriptions (age-washing: "she is 18" + a child's body)
                    "flatchest", "nobreast", "flatbody", "undevelop", "underdevelop", "smallframe",
                    "tinyframe", "tinybody", "smallbody", "littlegirl", "littleboy", "kleinerkorper",
                    "flachebrust", "keinebrust")

VIOLENCE_WORDS = {
    "gore", "gory", "blood", "bloody", "bleeding", "corpse", "mutilated", "torture", "decapitated",
    "beheaded", "murder", "massacre", "dismembered", "slaughter", "snuff", "rape", "raped", "rapist",
    "noncon", "necrophilia", "bestiality", "zoophilia", "blut", "leiche", "folter", "vergewaltigung",
}
VIOLENCE_SUBSTRINGS = ("gore", "mutilat", "decapitat", "dismember", "snuff", "noncon", "necrophil", "bestiality")

# This generator is sexual by design, so religious places/garments are blocked entirely
# (sexual content in a mosque or with a hijab is harassment, not art). Remove this set if you disagree.
RELIGIOUS_WORDS = {
    "mosque", "mosques", "moschee", "masjid", "hijab", "niqab", "burka", "burqa", "quran", "koran",
    "kaaba", "mecca", "church", "kirche", "synagogue", "synagoge", "cathedral", "kathedrale",
    "nun", "nonne", "priest", "pfarrer", "imam", "rabbi", "bible", "bibel",
}

# Numeric ages: "16 year old", "16yo", "16 y/o", "age 15", "16 jahre", "(14)"
_AGE_PATTERNS = [
    re.compile(r"\b(\d{1,2})\s*[- ]?\s*(?:yo|y/o|yr|yrs|year|years|yearold|jahre|jahr|j)\b"),
    re.compile(r"\b(?:age|aged|alter)\s*[:=]?\s*(\d{1,2})\b"),
]

# Forced hardening, appended server-side so the user cannot remove it.
FORCED_POSITIVE = "adult, mature adult woman or man, 25+ years old, fully grown"
FORCED_NEGATIVE = ("child, children, kid, minor, underage, teen, teenager, young, youth, loli, shota, baby, "
                   "toddler, schoolgirl, schoolboy, childlike, petite, flat chest, small body, "
                   "gore, blood, mutilation")


def _normalize(text):
    t = unicodedata.normalize("NFKD", text.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return t


def check_text(text, max_len=MAX_PROMPT_LEN):
    """Return None if OK, otherwise only the CATEGORY of the hit (never log the text itself)."""
    if not isinstance(text, str):
        return "invalid"
    if len(text) > max_len:
        return "length"

    raw = _normalize(text)

    # 1) numeric ages BEFORE leet translation (digits must stay digits)
    for pat in _AGE_PATTERNS:
        for m in pat.finditer(raw):
            if int(m.group(1)) < 18:
                return "minor"

    norm = raw.translate(_LEET)
    words = set(re.findall(r"[a-z]+", norm))
    collapsed = re.sub(r"[^a-z]", "", norm)

    if words & MINOR_WORDS or any(s in collapsed for s in MINOR_SUBSTRINGS):
        return "minor"
    if words & VIOLENCE_WORDS or any(s in collapsed for s in VIOLENCE_SUBSTRINGS):
        return "violence"
    if words & RELIGIOUS_WORDS:
        return "religious"
    return None


class Blocked(Exception):
    pass


def ensure_safe(*texts, max_len=MAX_PROMPT_LEN):
    """Raise Blocked if any text is not allowed. Use on user input AND on any LLM-generated text."""
    for t in texts:
        category = check_text(t or "", max_len)
        if category:
            print(f"[blocked] {category}", flush=True)  # category only, never the text
            raise Blocked(category)


def harden_prompts(prompt, negative_prompt):
    """Server-side: always push towards adults, always push away from minors."""
    return (f"{FORCED_POSITIVE}, {prompt}".strip(", "),
            f"{FORCED_NEGATIVE}, {negative_prompt or ''}".strip(", "))


# ---------------- Negative prompt guard ----------------
# A user could put "adult, mature" into the NEGATIVE prompt to cancel the forced adult push.
# So the negative prompt may not contain age-related words. (Gender words etc. stay allowed.)
NEGATIVE_BLOCK_WORDS = {"adult", "adults", "mature", "grown", "fullygrown", "old", "older", "elderly",
                        "aged", "milf", "erwachsen", "erwachsene", "reif"}


def check_negative(text, max_len=MAX_PROMPT_LEN):
    """None if OK, otherwise the category. Only age-cancelling words are blocked here."""
    if not isinstance(text, str):
        return "invalid"
    if len(text) > max_len:
        return "length"
    words = set(re.findall(r"[a-z]+", _normalize(text).translate(_LEET)))
    if words & NEGATIVE_BLOCK_WORDS:
        return "negative_adult"
    return None


def ensure_safe_negative(text, max_len=MAX_PROMPT_LEN):
    category = check_negative(text or "", max_len)
    if category:
        print(f"[blocked] {category}", flush=True)
        raise Blocked(category)


# ---------------- Image layer (best effort) ----------------
# Age classifier on the finished image. Weak without a face crop, and the "10-19" bucket also contains
# 18-19 year olds, so it is intentionally strict (false alarms possible).
_age_clf = None
_YOUNG_LABELS = {"0-2", "3-9", "10-19"}
IMAGE_THRESHOLD = 0.25


def _load_age_clf():
    global _age_clf
    if _age_clf is None:
        from transformers import pipeline
        _age_clf = pipeline("image-classification", model="nateraw/vit-age-classifier", device=-1)
    return _age_clf


def image_looks_young(image, fail_closed=True):
    """True -> block the image. If the classifier breaks and fail_closed=True, block as well."""
    try:
        results = _load_age_clf()(image.convert("RGB"), top_k=None)
        return any(r["label"] in _YOUNG_LABELS and r["score"] >= IMAGE_THRESHOLD for r in results)
    except Exception as e:
        print(f"[image check error] {type(e).__name__}", flush=True)
        return fail_closed
