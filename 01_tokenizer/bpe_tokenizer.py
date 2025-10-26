from collections import defaultdict
from pathlib import Path


def main():
    """Run the script."""
    dataset_path = "../data/shakespeare_dataset"

    my_tokenizer = BPETokenizer()
    my_tokenizer.train(dataset_path)


class BPETokenizer:
    """BPETokenizer class definition."""

    def __init__(self):
        """Initialize the class entity."""
        # encoding id -> token
        self.vocab = dict()
        # decoding token -> id
        self.reverse_vocab = dict()
        self.next_id = 0

    def _initialize_vocab(self, dataset_path):
        """Initialize vocab with individual characters from entire dataset."""
        chars = set()
        for file in dataset_path.iterdir():
            with open(file, "r") as f:
                chars.update(set(f.read()))

        chars = sorted(chars)
        self.vocab = {char: i for i, char in enumerate(chars)}
        self.reverse_vocab = {i: char for char, i in self.vocab.items()}
        self.next_id = len(self.vocab)

    def _tokenize_dataset(self, dataset_path):
        """Tokenize entire database."""
        # final list of tokenized lines
        tokenized_dataset = list()

        for file in dataset_path.iterdir():
            with open(file, "r") as f:
                text = f.read()

            # convert each vocab character to its token representation
            tokenized_text = [self.vocab[c] for c in text]
            # populate the final list with the new converted line
            tokenized_dataset.append(tokenized_text)

        return tokenized_dataset

    def _count_pairs(self, tokenized_dataset):
        """Count token pairs across entire dataset."""
        # initialize a dictionary for counting occurences
        count_pairs_dict = defaultdict(int)
        # max_count value
        max_count = 0

        for text in tokenized_dataset:
            for i in range(len(text) - 1):
                pair = (text[i], text[i + 1])
                # update count
                count_pairs_dict[pair] += 1
                # update max_count value
                pair_count = count_pairs_dict[pair]
                if pair_count > max_count:
                    max_count = pair_count

        return max_count, count_pairs_dict

    def _get_candidate_pair(self, max_count, count_pairs_dict):
        """Find a new pair for merge."""
        # find most frequent pairs
        popular_pairs = list()
        for pair, count in count_pairs_dict.items():
            # print(f"{self.reverse_vocab[pair[0]]}{self.reverse_vocab[pair[1]]}")
            # append popular_pairs list
            if count == max_count:
                popular_pairs.append(pair)

        candidate_pair = min(popular_pairs)

        print(f"candidate pair: {candidate_pair}")
        print(f"max count: {max_count}")

        return candidate_pair

    def _update_vocab(self, candidate_pair):
        # update vocab with new id-token pair
        self.vocab[candidate_pair] = self.next_id
        self.reverse_vocab[self.next_id] = candidate_pair

        self.next_id += 1

    def _merge_pair(self, tokenized_dataset, candidate_pair):
        """Update tokenized_dataset with the merged pair."""
        # update dataset
        check_max_count = 0
        for i in range(len(tokenized_dataset)):
            # original tokenized text
            text = tokenized_dataset[i]
            # updated tokenized text
            updated_text = list()

            # find paired tokens' position and populate new text
            idx = 0
            # -2 because idx starts from 0 + we need idx+1
            while idx <= len(text) - 2:
                if (text[idx], text[idx + 1]) == candidate_pair:
                    updated_text.append(self.vocab[candidate_pair])
                    idx += 1
                    check_max_count += 1
                else:
                    updated_text.append(text[idx])
                idx += 1

            # update text in dataset
            tokenized_dataset[i] = updated_text

        # check
        print(f"check_max_count: {check_max_count}")

        print(self.vocab)
        print(self.reverse_vocab)

        return tokenized_dataset

    def train(self, dataset: str):
        """Convert text to initial tokens."""
        dataset_path = Path(dataset)

        self._initialize_vocab(dataset_path)
        tokenized_dataset = self._tokenize_dataset(dataset_path)

        # get pairs information
        max_count, count_pairs_dict = self._count_pairs(tokenized_dataset)

        # get new candidate pair
        candidate_pair = self._get_candidate_pair(max_count, count_pairs_dict)

        # update vocab
        self._update_vocab(candidate_pair)

        # merge pair into a new token
        self._merge_pair(tokenized_dataset, candidate_pair)


if __name__ == "__main__":
    main()
