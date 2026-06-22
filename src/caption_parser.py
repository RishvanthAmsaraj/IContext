"""
Caption Parser: Extract objects from Florence-2 natural language descriptions.

Florence-2 generates descriptions like:
"There is a white desk in a room. There is a black computer on top of the desk.
There are two monitors on the desk in front of the laptop."

This module extracts the object nouns (desk, computer, monitors, laptop) so they
can be added to the YOLO detection list for a more complete scene understanding.
"""

import re


# Common English stop words and generic descriptors to filter out.
# Anything in this set will not be considered a meaningful object.
STOP_WORDS = {
    # Articles, determiners, pronouns
    "the", "a", "an", "this", "that", "these", "those", "some", "any", "all",
    "each", "every", "no", "its", "their", "his", "her", "my", "your", "our",
    # Verbs (common auxiliaries)
    "is", "are", "was", "were", "be", "been", "being", "has", "have", "had",
    "do", "does", "did", "can", "could", "will", "would", "should", "may",
    "might", "must", "shall", "sit", "sits", "stand", "stands", "rest", "rests",
    "appear", "appears", "seem", "seems", "look", "looks", "spin", "spins",
    "park", "parked", "sitting", "standing", "running", "playing", "working",
    # Prepositions
    "in", "on", "at", "to", "for", "of", "with", "by", "from", "up", "into",
    "through", "during", "before", "after", "above", "below", "between",
    "out", "against", "around", "among", "off", "over", "under", "onto",
    "upon", "within", "without", "across", "behind", "beside", "beyond",
    "near", "next", "outside", "inside", "along",
    # Conjunctions
    "and", "or", "but", "so", "because", "as", "until", "while", "if", "than",
    "that", "which", "who", "whom", "whose",
    # Adverbs
    "not", "only", "also", "just", "now", "then", "here", "there", "where",
    "when", "very", "too", "still", "already", "always", "never", "often",
    "sometimes", "usually", "really", "quite", "almost",
    # Generic spatial / size / time descriptors (not specific objects)
    "top", "bottom", "side", "left", "right", "front", "back", "middle",
    "center", "edge", "corner", "end", "beginning", "first", "second",
    "third", "last", "one", "two", "three", "four", "five", "six", "seven",
    "eight", "nine", "ten",
    # Common adjectives that aren't objects
    "large", "small", "big", "little", "long", "short", "high", "low",
    "wide", "narrow", "deep", "shallow", "old", "new", "good", "bad",
    "great", "white", "black", "red", "blue", "green", "yellow", "brown",
    "gray", "grey", "pink", "purple", "orange", "open", "closed", "full",
    "empty", "clean", "dirty", "dark", "bright", "lit",
    # Generic scene words (not specific objects)
    "room", "place", "area", "space", "background", "foreground", "image",
    "picture", "photo", "scene", "view", "way", "kind", "type", "sort",
    # Generic people words (we already detect "person" via YOLO)
    "man", "woman", "person", "people", "guy", "girl", "boy", "child",
    "someone", "anyone", "everyone", "nobody", "men", "women", "children",
    # Florence-2 specific filler words
    "there", "made", "make", "color", "colored", "them", "it", "its",
}


def _is_meaningful_object(word):
    """
    Check if a word is likely a meaningful object (not a stop word, too short, etc.)
    """
    word = word.strip(".,;:!?'\"").lower()

    # Too short or too long
    if len(word) < 3 or len(word) > 30:
        return False

    # Is a stop word or generic descriptor
    if word in STOP_WORDS:
        return False

    # Looks like a number
    if word.isdigit():
        return False

    # Ends with common verb/adjective suffixes (less likely to be a noun)
    # BUT we should be careful not to exclude real objects that end in -ing, etc.
    # Only exclude if the word is very common as a verb
    if word.endswith(("ness", "ment", "tion", "sion", "ous", "ive", "able", "ible")):
        return False

    return True


def extract_objects_from_caption(caption):
    """
    Extract object nouns from a Florence-2 caption.

    Strategy:
    1. Tokenize the caption into words
    2. Filter out stop words, numbers, and common non-noun words
    3. Keep only words that look like they could be objects
    4. Deduplicate and return

    Args:
        caption: Natural language string from Florence-2

    Returns:
        list of str: Unique object names found in the caption
    """
    if not caption:
        return []

    # Tokenize: split on whitespace and punctuation
    # Keep only alphabetic words
    words = re.findall(r"\b[a-zA-Z][a-zA-Z\-]*[a-zA-Z]\b|\b[a-zA-Z]\b", caption)

    # Filter and deduplicate while preserving order
    seen = set()
    objects = []
    for word in words:
        word_lower = word.lower()
        if _is_meaningful_object(word_lower) and word_lower not in seen:
            seen.add(word_lower)
            objects.append(word_lower)

    return objects


def merge_object_lists(yolo_objects, vlm_objects):
    """
    Combine objects from YOLO and VLM (Florence-2), removing obvious duplicates.

    Args:
        yolo_objects: List of object names from YOLO detection
        vlm_objects: List of object names extracted from VLM caption

    Returns:
        list of str: Combined unique list (VLM-only objects prefixed with "[VLM]")
    """
    # Normalize to lowercase
    yolo_set = set(obj.lower().strip() for obj in yolo_objects)
    vlm_set = set(obj.lower().strip() for obj in vlm_objects)

    # Find objects that VLM found but YOLO didn't
    novel = vlm_set - yolo_set

    # Combine: keep YOLO objects as-is, add novel VLM objects with prefix
    combined = list(yolo_objects)  # Keep original YOLO objects
    for obj in vlm_set:
        if obj in novel:
            combined.append(f"[VLM] {obj}")

    return combined


# Quick test
if __name__ == "__main__":
    test_captions = [
        "There is a white desk in a room. There is a black computer on top of the desk. There are two monitors on the desk in front of the laptop.",
        "A ceiling fan is spinning above the table. The room is lit by a lamp.",
        "There are three people standing near a building. A car is parked outside.",
        "A dog is sitting on a couch with a book on the table.",
        "Two women are sitting at a table with wine glasses and plates of food in front of them.",
        "A man is holding a tennis racket on a tennis court with a net.",
    ]

    for i, cap in enumerate(test_captions):
        print(f"\nCaption {i+1}: {cap}")
        objects = extract_objects_from_caption(cap)
        print(f"Extracted: {objects}")