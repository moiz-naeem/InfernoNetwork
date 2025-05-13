import nltk
from nltk import WordNetLemmatizer


nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download("averaged_perceptron_tagger_eng")
nltk.download("maxent_ne_chunker")
nltk.download("words")
nltk.download("maxent_ne_chunker_tab")
nltk.download("stopwords")


def extract_using_tag(tagged_pos, tag, p_o_s):
    lemmatizer = WordNetLemmatizer()
    words = []

    noise = {
        "d",
        "”",
        "“",
        "’",
        "‘",
        "s",
        "st",
        "er",
        "n",
        "th",
        "d.",
        "t",
        "en",
        "'",
        "ne",
        "e",
        "ll",
        "re",
        "o",
        "i",  # misclassified as both noun and adjective
        "whence",  # not a noun nor adjective
        "forth",  # not a noun nor adjective
        "thence",  # not a noun nor adjective
        "hence",  # not a noun nor adjective
        "aught",  # not a noun nor adjective
        "rous",  # part of a adjective e.g. thund'rous
        "such",  # not an adjective
        "spake",  # archaic for spoke, a verb
        "mine",  # not noun nor adj
        "ken",  # pronoun
    }

    for word, pos in tagged_pos:
        word = word.lower()
        if pos != tag or word in noise:
            continue
        lemma = lemmatizer.lemmatize(word, pos=p_o_s)

        words.append(lemma)
    return words


def read_clean_text(word_list):
    for i in range(len(word_list)):
        current_word = word_list[i]
        # strip all possbile endings
        cleaned_word: str = current_word.strip("“”;,.?!;:").lower()

        if cleaned_word == "thou" or cleaned_word == "thee":
            cleaned_word = "you"
        elif cleaned_word == "thine":
            cleaned_word = "yours"
        elif cleaned_word == "thy":
            cleaned_word = "your"
        elif cleaned_word == "ye":
            cleaned_word = "you"
        elif cleaned_word == "e’en":
            cleaned_word = "even"
        elif cleaned_word == "e’er":
            cleaned_word = "ever"
        elif cleaned_word == "o’er":
            cleaned_word = "over"
        elif cleaned_word == "heav’n":
            cleaned_word = "heaven"
        elif cleaned_word == "oft":
            cleaned_word = "often"
        elif cleaned_word == "hath":
            cleaned_word = "have"
        elif cleaned_word == "lo":
            cleaned_word = "look"
        elif cleaned_word == "doth":
            cleaned_word = "do"
        elif cleaned_word == "’gainst":
            cleaned_word = "against"
        elif cleaned_word.endswith("’d"):
            start = cleaned_word.split("’")[0]
            cleaned_word = start + "ed"

        # check case of the word
        if current_word[0].isupper():
            cleaned_word = cleaned_word.capitalize()

        # Add the original end to the word
        if "," in current_word:
            word_list[i] = cleaned_word + ","

        elif "!" in current_word:
            word_list[i] = cleaned_word + "!"

        elif "." in current_word:
            word_list[i] = cleaned_word + "."

        elif "?" in current_word:
            word_list[i] = cleaned_word + "?"

        elif ";" in current_word:
            word_list[i] = cleaned_word + ";"

        elif ":" in current_word:
            word_list[i] = cleaned_word + ":"

        else:
            word_list[i] = cleaned_word

    read = " ".join(word_list)

    return read


def process_text(file_path):
    lines = []
    # clean each line separately, lines are needed later
    with open(file_path, "r", encoding="utf-8") as text:
        for line in text:
            if line == "" or line == "\n" or line == " ":
                continue
            line = read_clean_text(line.split())
            lines.append(line)

    # join lines to tokenize
    read = "\n".join(lines)

    with open("cleaned.txt", "w") as new:
        new.write(read)
        new.close()

    all_word_token = nltk.tokenize.word_tokenize(read)

    words_token_tagged_by_pos = nltk.pos_tag(all_word_token)

    adjectives = extract_using_tag(words_token_tagged_by_pos, "JJ", "a")
    nouns = extract_using_tag(words_token_tagged_by_pos, "NN", "n")

    noun_frequencis = nltk.FreqDist(nouns)
    adj_frequencies = nltk.FreqDist(adjectives)

    top_100_nouns = [word for word, count in noun_frequencis.most_common(100)]
    top_100_adjs = [word for word, count in adj_frequencies.most_common(100)]

    return (lines, top_100_nouns, top_100_adjs)
