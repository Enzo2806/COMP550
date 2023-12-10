import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

class Preprocessor:
    def __init__(self):
        # Initialize tokenizers for the source and target languages
        self.source_tokenizer = None
        self.target_tokenizer = None

    @staticmethod
    def load_data(source_file_path, target_file_path):
        """
        Load data from two separate files where each line in one file corresponds to the line in the other file.
        """
        with open(source_file_path, 'r', encoding='utf-8') as file:
            source_lines = file.read().split('\n')
        with open(target_file_path, 'r', encoding='utf-8') as file:
            target_lines = file.read().split('\n')

        # Ensure both files have the same number of lines
        min_lines = min(len(source_lines), len(target_lines))
        return [(source_lines[i], target_lines[i]) for i in range(min_lines) if source_lines[i] and target_lines[i]]
    
    @staticmethod
    def clean_sentence(sentence):
        """
        Clean a sentence by lowering its case and removing non-alphanumeric characters.
        """
        sentence = sentence.lower().strip()
        sentence = re.sub(r'[^a-zA-Z0-9áéíñóúàèìòù]', ' ', sentence)
        return sentence

    def tokenize(self, sentences):
        """
        Tokenize a list of sentences and convert them to sequences of integers.
        This function updates the tokenizer for its corresponding instance.
        """
        tokenizer = Tokenizer(char_level=False)
        tokenizer.fit_on_texts(sentences)
        return tokenizer, tokenizer.texts_to_sequences(sentences)

    @staticmethod
    def pad(sequences, maxlen=None):
        """
        Pad sequences to ensure uniform length.
        """
        return pad_sequences(sequences, padding='post', maxlen=maxlen)

    def preprocess(self, source_file_path, target_file_path):
        """
        Preprocess the data from given source and target files. This includes loading, cleaning, tokenizing, padding, and splitting the dataset into
        training, validation, and testing sets.
        """
        # Load and clean the datasets
        data = self.load_data(source_file_path, target_file_path)
        data = [(self.clean_sentence(source), self.clean_sentence(target)) for source, target in data]

        # Tokenize
        source_sentences, target_sentences = zip(*data)
        self.source_tokenizer, source_sequences = self.tokenize(source_sentences)
        self.target_tokenizer, target_sequences = self.tokenize(target_sentences)

        # Pad sequences
        source_sequences = self.pad(source_sequences)
        target_sequences = self.pad(target_sequences, maxlen=len(source_sequences[0]))

        # First split data to get 80% for training and 20% for validation + test
        source_train, source_temp, target_train, target_temp = train_test_split(
            source_sequences, target_sequences, test_size=0.2, random_state=42)

        # Then split the 20% into half to get 10% for validation and 10% for testing
        source_val, source_test, target_val, target_test = train_test_split(
            source_temp, target_temp, test_size=0.5, random_state=42)

        # Return the training, validation, and testing sets
        return source_train, source_val, source_test, target_train, target_val, target_test
