from matplotlib import pyplot as plt
import nltk
from req_1_text_processing import extract_using_tag, read_clean_text


def relation_evolution(file_path, top_nouns, top_adjs):
    """
    Tracks the cumulative evolution of noun-noun, noun-adjective, and adjective-adjective relationships
    across the lines of a text document.

    Parameters:
        file_path (str): Path to the text file.
        top_nouns (set or list): Frequently occurring nouns to track.
        top_adjs (set or list): Frequently occurring adjectives to track.

    Saves:
        'relationship_evolution.png': Line plot of how each relationship type evolves.
    """
    lines = []
    noun_noun, noun_adj, adj_adj = [], [], []

    # Read and clean lines from file
    with open(file_path, "r", encoding="utf-8") as text:
        for line in text:
            cleaned_line = read_clean_text(line.split())
            if cleaned_line:
                lines.append(cleaned_line)

    for i, line in enumerate(lines):
        adj_count, noun_count = 0, 0
        tokens = nltk.word_tokenize(line)
        tagged_tokens = nltk.pos_tag(tokens)

        # Extract relevant words
        adjectives = extract_using_tag(tagged_tokens, "JJ", "a")
        nouns = extract_using_tag(tagged_tokens, "NN", "n")

        # Count matching words
        adj_count = sum(1 for adj in adjectives if adj in top_adjs)
        noun_count = sum(1 for noun in nouns if noun in top_nouns)

        # Compute relationships
        noun_noun_val = max(0, noun_count * (noun_count - 1) / 2)
        noun_adj_val = noun_count * adj_count
        adj_adj_val = max(0, adj_count * (adj_count - 1) / 2)

        # Cumulative sum
        noun_noun.append(noun_noun_val + (noun_noun[i - 1] if i > 0 else 0))
        noun_adj.append(noun_adj_val + (noun_adj[i - 1] if i > 0 else 0))
        adj_adj.append(adj_adj_val + (adj_adj[i - 1] if i > 0 else 0))

    # Plotting
    plt.figure(figsize=(15, 15))

    # Create 3 square-shaped subplots
    plt.subplot(3, 3, 1)  # Row 1, Column 1
    plt.plot(noun_noun, "r-", linewidth=1.5)
    plt.title("Noun-Noun Relationships", fontsize=14)
    plt.ylabel("Cumulative Count")
    plt.grid(True, alpha=0.3)

    plt.subplot(3, 3, 4)  # Row 2, Column 1
    plt.plot(noun_adj, "g-", linewidth=1.5)
    plt.title("Noun-Adjective Relationships", fontsize=14)
    plt.ylabel("Cumulative Count")
    plt.grid(True, alpha=0.3)

    plt.subplot(3, 3, 7)  # Row 3, Column 1
    plt.plot(adj_adj, "b-", linewidth=1.5)
    plt.title("Adjective-Adjective Relationships", fontsize=14)
    plt.xlabel("Line Number")
    plt.ylabel("Cumulative Count")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("relationship_evolution.png", dpi=300, bbox_inches="tight")
    plt.close()
