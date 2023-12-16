import torch
import torch.nn as nn
import math

# Loading dataset imports
from torch.utils.data import Dataset, DataLoader # for creating the dataloader
import json # for loading the json file
from TranslationDataset import TranslationDataset # the custom dataset class


# Training imports
from Transformer_model import Transformer, build_transformer # the model
from torch.utils.tensorboard import SummaryWriter  # for logging during training
from tqdm import tqdm # for the progress bar during training


# Load the JSON header file
def load_json_header(json_file):
    with open(json_file) as json_data:
        d = json.load(json_data)
        return d

config = load_json_header('config.json')


# Load the datasets
# Get the dataset path from the config file
en_it_dataset_path = config['en-it-save-path']

# Load the dataset
en_it_train = torch.load(en_it_dataset_path + 'train_ds.pt')
en_it_val = torch.load(en_it_dataset_path + 'val_ds.pt')
en_it_test = torch.load(en_it_dataset_path + 'test_ds.pt')

# Load the vocabularies from the config file
source_vocab = torch.load(en_it_dataset_path + 'source_vocab.pt')
target_vocab = torch.load(en_it_dataset_path + 'target_vocab.pt')

# Print the size of the dataset
print('Size of training dataset: ', len(en_it_train))
print('Size of validation dataset: ', len(en_it_val))
print('Size of test dataset: ', len(en_it_test))

# Create dataloaders
train_dl = DataLoader(en_it_train, batch_size=config["batch_size"], shuffle=True)
val_dl = DataLoader(en_it_val, batch_size=1, shuffle=False)
test_dl = DataLoader(en_it_test, batch_size=config["batch_size"], shuffle=False)


# Select device: cuda, mps or cpu
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print('Device:', device)

def train(hyperparams):
    # Define the model
    model = build_transformer(
                        len(source_vocab),
                        len(target_vocab),
                        src_seq_len= config["max_seq_len"],
                        trg_seq_len= config["max_seq_len"],
                        d_model = 512,
                        N = 1,
                        h = 4,
                        dropout = 0.1,
                        d_ff = 2048).to(device)
    writer = SummaryWriter()

    # Define the hyperparameters from the given dictionary
    lr = hyperparams['lr']
    epochs = hyperparams['epochs']

    # Define the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams['lr'], eps=1e-9)

    # Define the loss function
    # Ignore the padding token, which has index 3 in the vocabulary (see function build_vocab in Preprocessing.ipynb file)
    loss_fn = nn.CrossEntropyLoss(ignore_index=3, label_smoothing=0.1).to(device)

    step = 0 # for logging the loss

    for epoch in range (epochs):
        torch.mps.empty_cache() # empty the cache
        model.train()
        iter = tqdm(train_dl, desc=f'Epoch {epoch}')
        for batch in iter:
            encoder_input = batch['encoder_input'].to(device) # size (batch_size, seq_len)
            decoder_input = batch['decoder_input'].to(device) # size (batch_size, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # size (batch_size, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device) # size (batch_size, 1, seq_len, seq_len)
            label = batch['label'].to(device) # size (batch_size, seq_len)

            # Run the tensors through the model
            encoder_output = model.encode(encoder_input, encoder_mask)  # size (batch_size, seq_len, d_model)
            decoder_output = model.decode(decoder_input, encoder_output, encoder_mask, decoder_mask) # size (batch_size, seq_len, d_model)
            output = model.output(decoder_output) # size (batch_size, seq_len, trg_vocab_size)

            # Calculate the loss
            # Flatten the output and label tensors to size (batch_size * seq_len, trg_vocab_size)
            loss = loss_fn(output.view(-1, len(target_vocab)), label.view(-1))
            iter.set_postfix(loss=loss.item()) # print the loss
            writer.add_scalar('Loss/Step', loss.item(), step) # log the loss
            writer.flush()

            # Backpropagation
            loss.backward()    

            # Update the parameters
            optimizer.step()
            optimizer.zero_grad()

            step += 1


hyperparameters = {
    'lr': 0.0001,
    'epochs': 1
}

train(hyperparameters)