# IMPORTS
import torch
import torch.nn as nn
import json # for loading the json file
from torch.utils.data import Dataset # for creating the dataset

# for tokenizing the sentences
import nltk 
from nltk.tokenize import word_tokenize 
nltk.download('punkt') 

# For splitting the dataset into train, validation anmd testing
import random
from sklearn.model_selection import train_test_split

from TranslationDataset import TranslationDataset


# Load the JSON header file
def load_json_header(json_file):
    with open(json_file) as json_data:
        d = json.load(json_data)
        return d

config = load_json_header('config.json')


def load_data(source_file_path, target_file_path):
    """
    Load data from two separate files where each line in one file corresponds to the line in the other file.
    """
    with open(source_file_path, 'r', encoding='utf-8') as file:
        source = file.read().split('\n')
    with open(target_file_path, 'r', encoding='utf-8') as file:
        target = file.read().split('\n')

    return source, target



def build_vocab(sentences, min_frequency=2, special_tokens=["[UNK]", "[SOS]", "[EOS]", "[PAD]"]):
    vocab = {}
    word_counts = {}

    # Initialize vocab with special tokens
    for token in special_tokens:
        vocab[token] = len(vocab)

    # Count word frequencies
    for sentence in sentences:
        for word in word_tokenize(sentence):
            word_counts[word] = word_counts.get(word, 0) + 1

    # Print some examples from the imported dataset
    print("Some tokenized examples from the imported dataset:")
    for i in range(5):
        print(word_tokenize(sentences[i]))
        
    # Add words above min frequency to vocab
    for word, count in word_counts.items():
        if count >= min_frequency:
            vocab[word] = len(vocab)

    return vocab




def shuffle_and_split(source_sentences, target_sentences, test_size, val_size):
    combined = list(zip(source_sentences, target_sentences))
    random.shuffle(combined)
    shuffled_source_sentences, shuffled_target_sentences = zip(*combined)

    # Splitting into train and test
    src_train_val, src_test, trg_train_val, trg_test = train_test_split(
        shuffled_source_sentences, shuffled_target_sentences, test_size = test_size, random_state =42)
    
    # Splitting train_val into train and val
    src_train, src_val, trg_train, trg_val = train_test_split(
        src_train_val, trg_train_val, test_size = val_size/(1 - test_size), random_state=42)

    return src_train, src_val, src_test, trg_train, trg_val, trg_test


def preprocess_datasets(source_file_path, target_file_path, save_path):
    """
    Preprocess the datasets and save them to files.
    """
    
    # Load data from source and target files
    source, target = load_data(source_file_path, target_file_path)

    # Check if source and target files have the same number of lines
    if len(source) != len(target):
        raise Exception("Source and target files do not have the same number of examples.")

    # Build vocab dictionaries for source and target languages
    source_vocab = build_vocab(source)
    target_vocab = build_vocab(target)

    # We first get the validation and testing set sizes located in the config file
    # We then calculate the training set size from them and the total dataset size (see the shuffle and split function above)
    val_size = config["val_size"]
    test_size = config["test_size"]

    # Split into train and validation sets, making sure that the source and target sentences are aligned
    src_train, src_val, src_test, trg_train, trg_val, trg_test = shuffle_and_split(source, target, test_size=test_size, val_size=val_size)

    # Check that the source and target datasets have the same size after split, otherwise raise an exception
    if len(src_train) != len(trg_train) or len(src_val) != len(trg_val) or len(src_test) != len(trg_test):
        raise Exception("Source and target datasets do not have the same size")
    
    # Get the maximum sequence length from the config file
    max_seq_len = config["max_seq_len"]
    
    # Create datasets
    # We use the TranslationDataset class defined above
    # We pass the source and target datasets, source and target languages, source and target vocabularies, and the sequence length (in the config file)
    train_ds = TranslationDataset(src_train, trg_train, source_vocab, target_vocab, max_seq_len)
    val_ds = TranslationDataset(src_val, trg_val, source_vocab, target_vocab, max_seq_len)
    test_ds = TranslationDataset(src_test, trg_test, source_vocab, target_vocab, max_seq_len)

    # Print some examples from the dataset print("Some examples from the dataset:")
    for i in range(1):
        print(train_ds[i])
        
    # Print dataset sizes and a sample from the training dataset
    print("Train dataset size:", len(train_ds))
    print("Validation dataset size:", len(val_ds))
    print("Test dataset size:", len(test_ds))

    # Save the datasets to files
    torch.save(train_ds, save_path + "train_ds.pt")
    torch.save(val_ds, save_path + "val_ds.pt")
    torch.save(test_ds, save_path + "test_ds.pt")

    # Save the vocabularies to files
    torch.save(source_vocab, save_path + "source_vocab.pt")
    torch.save(target_vocab, save_path + "target_vocab.pt")

# Main 
if __name__ == "__main__":
    # Preprocess the datasets
    print("Preprocessing english to italian dataset:")
    preprocess_datasets(config['en-it-dataset-english'] , config['en-it-dataset-italian'], config['en-it-save-path'])

    print()
    print("Preprocessing english to spanish dataset:")
    preprocess_datasets(config['en-es-dataset-english'], config['en-es-dataset-spanish'], config['en-es-save-path'])