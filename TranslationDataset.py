import torch
import torch.nn as nn
from torch.utils.data import Dataset
# for tokenizing the sentences
import nltk 
nltk.download('punkt') 

class TranslationDataset(Dataset):

    def __init__(self, source_ds, target_ds, src_vocab, trg_vocab, seq_len):
        super().__init__()
        self.source_ds = source_ds
        self.target_ds = target_ds
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.seq_len = seq_len
        self.sos_token = torch.tensor([src_vocab["[SOS]"]], dtype=torch.int64)
        self.eos_token = torch.tensor([src_vocab["[EOS]"]], dtype=torch.int64)
        self.pad_token = torch.tensor([src_vocab["[PAD]"]], dtype=torch.int64)
    
    def __len__(self):
        return len(self.source_ds)

    def tokenize_and_numericalize(self, sentence, vocab):
        return [vocab.get(word, vocab["[UNK]"]) for word in nltk.word_tokenize(sentence)]

    def causal_mask(self, seq_len):
        """
        Causal mask: each word in the decoder can only look at previous words
        This is done to prevent the decoder from looking at future words.
        """
        # Create a matrix of size seq_len x seq_len
        # Fill the upper triangle with 0s and lower triangle with 1s
        # This is done to prevent the decoder from looking at future words
        return torch.tril(torch.ones((1, seq_len, seq_len), dtype=torch.int64)) 

    def __getitem__(self, idx):
        # Get source and target sentences
        src = self.source_ds[idx]
        trg = self.target_ds[idx]

        # Cnvert text to tokens to split into single words
        # Convert to input ids: map words to vocab ids
        enc_input_tokens = self.tokenize_and_numericalize(src, self.src_vocab)
        dec_input_tokens = self.tokenize_and_numericalize(trg, self.trg_vocab)

        # PAD input sequences to the same length so that model can be trained in batches where each input sequence has the same length
        # minus 2: remove SOS and EOS tokens from the count
        enc_num_padding = self.seq_len - len(enc_input_tokens) - 2
        
        # Only minus 1: remove SOS token from the count (EOS token is not included in the input sequence)
        dec_num_padding = self.seq_len - len(dec_input_tokens) - 1
        
        if enc_num_padding < 0 or dec_num_padding < 0:
            raise Exception("Sequence length is too long")

        # Add SOS and EOS tokens to the input sequence
        encoder_input = torch.cat([
            self.sos_token, 
            torch.tensor(enc_input_tokens, dtype = torch.int64),
            self.eos_token, 
            self.pad_token.repeat(enc_num_padding)])

        # Add SOS token to the decoder input sequence
        decoder_input = torch.cat([
            self.sos_token, 
            torch.tensor(dec_input_tokens, dtype = torch.int64),
            self.pad_token.repeat(dec_num_padding)])
        
        # Add EOS token to the decoder label sequence
        label = torch.cat([
            torch.tensor(dec_input_tokens, dtype = torch.int64),
            self.eos_token, 
            self.pad_token.repeat(dec_num_padding)])

        if encoder_input.shape[0] != self.seq_len:
            raise Exception("Encoder input sequence length is not correct, expected {}, got {}".format(self.seq_len, encoder_input.shape[0]))
        if decoder_input.shape[0] != self.seq_len:
            raise Exception("Decoder input sequence length is not correct, expected {}, got {}".format(self.seq_len, decoder_input.shape[0]))
        if label.shape[0] != self.seq_len:
            raise Exception("Label sequence length is not correct, expected {}, got {}".format(self.seq_len, label.shape[0]))

        return {
            "encoder_input": encoder_input, # size seq_len
            "decoder_input": decoder_input, # size seq_len
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1,1, Seq_len) # DOnt want the padding tokens to participate in the self attention mechnaism
            # Causal mask: each word can onlky look at previosu words and not future words and padding tokens as well. 
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & self.causal_mask(decoder_input.size(0)), # (1,1, Seq_len) & (1, Seq_len, Seq_len)
            "label": label,
            "src": src,
            "trg": trg
        }