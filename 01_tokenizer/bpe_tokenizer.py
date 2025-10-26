from collections import defaultdict
from pathlib import Path


def main():
    """Run the script."""
    text = Path("../data/text1.txt")

    my_tokenizer = BPETokenizer()
    my_tokenizer.train(text)


class BPETokenizer:
    """BPETokenizer class definition."""

    def __init__(self):
        """Initialize the class entity."""
        # encoding id -> token
        self.vocab = dict()
        # decoding token -> id
        self.reverse_vocab = dict()
        # (token1, token2): merged_token
        self.merges = dict()
        self.next_id = 0

    def _initialize_vocab(self, text):
        """Start with individual characters."""
        with open(text, "r") as f:
            chars = sorted(set(f.read()))

        self.vocab = {char: i for i, char in enumerate(chars)}
        self.reverse_vocab = {i: char for char, i in self.vocab.items()}
        self.next_id = len(self.vocab)

    def _tokenize_text(self, text):
        """Prepare text database for token representation."""
        with open(text, "r") as f:
            lines = f.readlines()

        # final list of tokenized lines
        tokenized_text = list()

        # convert each vocab character to its token representation
        for line in lines:
            # each line is the list of tokens
            tokenized_line = [self.vocab[c] for c in line]
            # populate the final list with the new converted line
            tokenized_text.append(tokenized_line)

        return tokenized_text

    def _count_pairs(self, tokenized_text):
        """Count token pairs across entire database."""
        # initialize a dictionary for counting occurences
        count_pairs_dict = defaultdict(int)
        # max_count value
        max_count = 0

        for line in tokenized_text:
            for i in range(len(line) - 1):
                pair = (line[i], line[i + 1])
                # update count
                count_pairs_dict[pair] += 1
                # update max_count value
                if count_pairs_dict[pair] > max_count:
                    max_count = count_pairs_dict[pair]

        return max_count, count_pairs_dict

    def _merge_pair(self, tokenized_text, merge_pair):
        # update tokenized text
        for i in range(len(tokenized_text)):
            line = tokenized_text[i]
            # print("1", tokenized_text[i])

            # updated list of tokens
            updated_line = list()

            # len line - 2 for staying within range
            len_line = len(line) - 2

            # keep empty lines as is
            if len_line == 0:
                updated_line.append(line)
                continue

            # find paired token's position and populate new line
            idx = 0
            while idx <= len_line:
                if (line[idx], line[idx + 1]) == merge_pair:
                    updated_line.append(self.merges[merge_pair])
                    idx += 1
                else:
                    updated_line.append(line[idx])
                idx += 1
            tokenized_text[i] = updated_line
            # print("2", tokenized_text[i])

        print(tokenized_text[6])

        return tokenized_text

    def train(self, text):
        """Convert text to initial tokens."""
        self._initialize_vocab(text)
        tokenized_text = self._tokenize_text(text)

        # get pairs information
        max_count, count_pairs_dict = self._count_pairs(tokenized_text)

        popular_pairs = list()
        for pair, count in count_pairs_dict.items():
            # print(f"{self.reverse_vocab[pair[0]]}{self.reverse_vocab[pair[1]]}")
            if count == max_count:
                popular_pairs.append(pair)

        # for pair in popular_pairs:
        #    print(self.reverse_vocab[pair[0]], self.reverse_vocab[pair[1]])

        merge_pair = min(popular_pairs)
        self.merges[merge_pair] = self.next_id
        self.next_id += 1


if __name__ == "__main__":
    main()
