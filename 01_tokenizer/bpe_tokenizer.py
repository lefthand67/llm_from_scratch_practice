from pathlib import Path


def main():
    """Run the script."""
    text = Path("../data/text1.txt")

    my_tokenizer = BPETokenizer()
    my_tokenizer.train(text)

    print(my_tokenizer.vocab)
    print(my_tokenizer.reverse_vocab)
    # print(my_tokenizer._tokenize_text(text))


class BPETokenizer:
    """BPETokenizer class definition."""

    def __init__(self):
        """Initialize the class entity."""
        # token: id
        self.vocab = dict()
        # for decoding
        self.reverse_vocab = dict()
        # (token1, token2): merged_token
        self.merges = dict()

    def _initialize_vocab(self, text):
        """Start with individual characters."""
        with open(text, "r") as f:
            chars = sorted(set(f.read()))

        self.vocab = {char: i for i, char in enumerate(chars)}
        self.reverse_vocab = {i: char for char, i in self.vocab.items()}

    def _tokenize_text(self, text):
        """Prepare text database for token representation."""
        tokenized_text = list()

        with open(text, "r") as f:
            lines = f.readlines()
            for line in lines:
                tokenized_line = []
                for c in line:
                    tokenized_line.append(self.vocab[c])
                tokenized_text.append(tokenized_line)

        return tokenized_text

    def train(self, text):
        """Convert text to initial tokens."""
        self._initialize_vocab(text)
        self._tokenize_text(text)


if __name__ == "__main__":
    main()
