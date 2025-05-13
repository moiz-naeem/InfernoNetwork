from matplotlib import pyplot as plt
import nltk
from req_1_text_processing import extract_using_tag, read_clean_text


def relation_evolution(file_path, top_nouns, top_adjs):
    lines = []

    noun_noun = []
    noun_adj = []
    adj_adj = []

    with open(file_path, "r", encoding="utf-8") as text:
        for line in text:
            line = read_clean_text(line.split())
            if line == "":
                continue
            lines.append(line)

    for i in range(len(lines)):
        adj_this_line = 0
        noun_this_line = 0

        current_line = lines[i]
        all_word_token = nltk.tokenize.word_tokenize(current_line)

        words_token_tagged_by_pos = nltk.pos_tag(all_word_token)

        adjectives = extract_using_tag(words_token_tagged_by_pos, "JJ", "a")
        nouns = extract_using_tag(words_token_tagged_by_pos, "NN", "n")

        for adj in adjectives:
            if adj in top_adjs:
                adj_this_line += 1

        for noun in nouns:
            if noun in top_nouns:
                noun_this_line += 1

        # relationships on this line
        noun_noun.append(max(0, noun_this_line * (noun_this_line - 1) / 2))
        noun_adj.append(noun_this_line * adj_this_line)
        adj_adj.append(max(0, adj_this_line * (adj_this_line - 1) / 2))

        if i != 0:
            noun_noun[i] += noun_noun[(i - 1)]
            noun_adj[i] += noun_adj[(i - 1)]
            adj_adj[i] += adj_adj[(i - 1)]

    plt.figure(figsize=(12, 12))
    plt.plot(range(len(lines)), noun_noun, "r-", linewidth=1.5)
    plt.title("Noun-Noun Relationships", fontsize=16)
    plt.ylabel("Number of Relationships")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("noun_noun_relationships.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(12, 12))
    plt.plot(range(len(lines)), noun_adj, "g-", linewidth=1.5)
    plt.title("Noun-Adjective Relationships", fontsize=16)
    plt.ylabel("Number of Relationships")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("noun_adj_relationships.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(12, 12))
    plt.plot(range(len(lines)), adj_adj, "b-", linewidth=1.5)
    plt.title("Adjective-Adjective Relationships", fontsize=16)
    plt.xlabel("Line Number")
    plt.ylabel("Number of Relationships")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("adj_adj_relationships.png", dpi=300, bbox_inches="tight")
    plt.close()
