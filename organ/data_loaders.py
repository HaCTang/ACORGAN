import numpy as np
from re import compile as _Re


class Gen_Dataloader():
    """Data iterator to load sample."""
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def create_batches(self, samples):
        self.num_batch = int(len(samples) / self.batch_size)
        samples = samples[:self.num_batch * self.batch_size]
        # self.sequence_batch = np.split(np.array(samples), self.num_batch, 0)
        self.sequence_batch = [samples[i:i + self.batch_size] 
                                for i in range(0, len(samples), self.batch_size)]
        self.pointer = 0

    def next_batch(self):
        ret = self.sequence_batch[self.pointer]
        self.pointer = (self.pointer + 1) % self.num_batch
        # return ret
        return [[item[0], item[1]] for item in ret]  # Return a 3D list with strings and labels separately
        
    def reset_pointer(self):
        self.pointer = 0


class Dis_Dataloader():
    def __init__(self):
        self.vocab_size = 5000

    def load_data_and_labels(self, positive_examples, negative_examples):
        """
        Loads MR polarity data from files, splits the data into words and generates labels.
        Returns split sentences and labels.
        """
        # Split by words
        x_text = positive_examples + negative_examples

        # Generate labels
        positive_labels = [[0, 1] for _ in positive_examples]
        negative_labels = [[1, 0] for _ in negative_examples]
        y = np.concatenate([positive_labels, negative_labels], 0)

        return [x_text, y]

    def load_train_data(self, positive_file, negative_file):
        """
        Returns input vectors, labels, vocabulary, and inverse vocabulary.
        """
        # Load and preprocess data
        classified_sentences, labels = self.load_data_and_labels(
            positive_file, negative_file)
        combined = list(zip(classified_sentences, labels))
        np.random.shuffle(combined)
        x_shuffled, y_shuffled = zip(*combined)
        x_shuffled = list(x_shuffled)
        y_shuffled = list(y_shuffled)
        self.sequence_length = 20 
        return [x_shuffled, y_shuffled]

    def load_test_data(self, positive_file, test_file):
        test_examples = []
        test_labels = []
        with open(test_file)as fin:
            for line in fin:
                line = line.strip()
                line = line.split()
                parse_line = [int(x) for x in line]
                test_examples.append(parse_line)
                test_labels.append([1, 0])

        with open(positive_file)as fin:
            for line in fin:
                line = line.strip()
                line = line.split()
                parse_line = [int(x) for x in line]
                test_examples.append(parse_line)
                test_labels.append([0, 1])

        test_examples = np.array(test_examples)
        test_labels = np.array(test_labels)
        shuffle_indices = np.random.permutation(np.arange(len(test_labels)))
        x_dev = test_examples[shuffle_indices]
        y_dev = test_labels[shuffle_indices]

        return [x_dev, y_dev]

    def batch_iter(self, data, batch_size, num_epochs):
        """
        Generates a batch iterator for a dataset.
        """
        data = list(data)  # Convert zip object to list to get its length
        data_size = len(data)
        num_batches_per_epoch = int(len(data) / batch_size) + 1
        for epoch in range(num_epochs):
            # Shuffle the data at each epoch
            np.random.shuffle(data)
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                yield data[start_index:end_index]