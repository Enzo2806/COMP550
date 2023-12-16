import math
import torch
import torch.nn as nn

"""

 This is the Transformer code explained in the original Transformer paper (https://arxiv.org/abs/1706.03762) by Vaswani et al.

 
 """

# Input embeddings : convert input sentence to vectors of size d_model. 
# This is done by multiplying the input sentence with a weight matrix.
# Note: in the embeddings layers, we multiply the input by sqrt(d_model) to prevent the gradients from exploding.
class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        # Embedding layer : convert input IDs to vectors
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        # Multiply the input by sqrt(d_model) to prevent the gradients from exploding
        return self.embedding(x) * math.sqrt(self.d_model)


# Positional encoding : add information about the position of each word in the sentence. 
# This is done by adding another vector of same size as the embeddings (512 here) to the input embeddings.
# We don't follow the formula from the paper which was: PE(pos, 2i) = sin(pos/10000^(2i/d_model)) and PE(pos, 2i+1) = cos(pos/10000^(2i/d_model))
# Instead, we use the formula using log space for more numerical stability: 
class PositionalEncoding(nn.Module):
    def __init__(self, d_model:int, seq_len_:int, dropout:float):
        super().__init__()
        self.d_model = d_model
        self.seq_len_ = seq_len_
        self.dropout = nn.Dropout(dropout)

        # Create a matrix of shape (seq_len, d_model): we need vectors of d_model size but we need seq_lentgh number of them
        # The matrix contains the positional encoding for each position in the sequence
        pe = torch.zeros(seq_len_, d_model)

        # Create a vector of shape (seq_len, 1) containing the positions of each word in the sequence
        position = torch.arange(0, seq_len_, dtype=torch.float).unsqueeze(1)
        # Create a vector of shape (1, d_model) containing the values of 1/10000^(2i/d_model) for each i
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # Apply the sin to even indices in the positional encoding matrix
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply the cos to odd indices in the positional encoding matrix
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add a batch dimension to the positional encoding matrix: (seq_len, d_model) -> (1, seq_len, d_model)
        pe = pe.unsqueeze(0)

        # Register the positional encoding matrix as a buffer (a buffer is a tensor that is not a model parameter). 
        # This is because we want that tensor to be saved along with the model when we save it.
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add the positional encoding to the input embeddings
        # The positional encoding matrix has shape (1, seq_len, d_model) but the input embeddings have shape (batch_size, seq_len, d_model)
        # So we need to add a batch dimension to the positional encoding matrix
        x = x + (self.pe[:, :x.size(1), :]).requires_grad_(False) 
        return self.dropout(x)
    

# Layer normalization (Norm in the Add & Norm layer) : Batch of N items
# We want to normalize each item in the batch separately. We do this by computing the mean and variance of each item in the batch. 
# We then normalize each item by subtracting the mean and dividing by the variance.
# We then multiply each item by a learnable parameter gamma and add a learnable parameter beta.
# The model learns the values of gamma and beta during training.
class LayerNorm(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps # epsilon is a small number used to prevent division by zero / numerical instability
        
        # Create two learnable parameters: gamma and beta
        self.gamma = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # Compute the standard deviation and mean of each item in the batch
        mean = x.mean(-1, keepdim=True) # We keep the dimension to be able to subtract the mean from each item in the batch
        std = x.std(-1, keepdim=True)

        # Normalize each item in the batch 
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


# Feed forward layer : a simple feed forward neural network with two linear transformations and a ReLU activation function
# The input is a batch of N items, each item has a dimension of d_model
# The inner layer has dimension 2048
# The outer layer has dimension d_model
class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # Linear transformation 1: W_1x + b_1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # Linear transformation 2: W_2x + b_2

    def forward(self, x):
        # The input is of shape (Batch, seq_len, d_model)
        # Apply the first linear transformation and the ReLU activation function --> (Batch, seq_len, d_ff)
        x = nn.functional.relu(self.linear_1(x))
        x = self.dropout(x)
        # Apply the second linear transformation --> (Batch, seq_len, d_model)
        x = self.linear_2(x)
        return x
    
# Multi-head attention : 
# Input is first copied into 3 matrices: Q, K, V
# Each matrix has shape (Batch, seq_len, d_model)
# We then multply Q, K, V by 3 different weight matrices to get Q', K', V' of shape (Batch, seq_len, d_model)
# We then split each matrix into h heads of shape (Batch, seq_len, d_model/h). Each head has access to a different representation of the input sentences with a different embedding of the words.
# We then compute the scaled dot product attention for each head separately
#   head_i = Attention(Q'W_i^Q, K'W_i^K, V'W_i^V) where W_i^Q, W_i^K, W_i^V are weight matrices of shape (d_model, d_model/h)
# We then concatenate the h heads into a single matrix of shape (Batch, seq_len, d_model) and multiply it by a weight matrix to get the output of the multi-head attention
class MutliHeadAttention(nn.Module):
    def __init__(self, d_model: int, h:int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.h = h
        # CHeck that d_model is divisible by h
        if not d_model % h == 0:
            raise ValueError(f'd_model ({d_model}) must be divisible by h ({h})')
        self.d_k = d_model // h # d_k is the dimension of each head (d_model/h)

        # Create the weight matrices for Q, K, V
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)

        # Create the weight matrix for the output of the multi-head attention
        # In the paper, its shape is (h*dv, d_model) where dv is equal to d_k (d_model/h) so its shape is (d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    

    def forward(self, Q, K, V, mask): 
        # Q, K, V have shape (Batch, seq_len, d_model)
        # The mask is of shape (Batch, d_model, d_model)
        # We therefore get an output of shape (Batch, seq_len, d_model)
        query = self.W_Q(Q)
        key = self.W_K(K)
        value = self.W_V(V)

        # Split each matrix into h heads of shape (Batch, seq_len, d_k)
        # We do this by splitting the last dimension of each matrix into h dimensions (we split the embedding of each word into h parts not the sentence)
        # We then transpose the result to get the shape (Batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        x, self.attention_scores = MutliHeadAttention.attention(query, key, value, mask, self.dropout)

        # Concatenate the h heads into a single matrix of shape (Batch, seq_len, d_model)
        # We transpose because we went from (Batch, h, seq_len, d_k) to (Batch, seq_len, h, d_k) by computing the attention for each head separately
        # Contiguous() is used to make sure that the memory is laid out in a contiguous chunk (this is needed for the reshape in pytorch)
        # We then reshape the matrix to get the shape (Batch, seq_len, h*d_k) = (Batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], x.shape[2], self.h * self.d_k)

        # Apply the last linear transformation to get the output of the multi-head attention
        # The output has shape (Batch, seq_len, d_model)
        return self.W_O(x)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]

        # Compute the scaled dot product attention
        # The attention scores have shape (Batch, h, seq_len, seq_len)
        # We transpose the last two dimensions to go from (Batch, h, seq_len, d_k) to (Batch, h, d_k, seq_len) for the key matrix
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        # The mask is used if we want some words to not interact with other words (e.g. padding tokens)
        # We replace the attention scores with small values for the masked words so that they don't affect the other words before applying the softmax
        # We do this by setting the attention score to -inf for the masked words
        # The softmax will then assign a probability of 0 to the masked words
        # The masked words will then have no effect on the other words
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e-8) 
 
        # Apply the softmax to get the attention probabilities
        # The attention probabilities have shape (Batch, h, seq_len, seq_len)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1) # dim=-1 means that we apply the softmax to the last dimension
        
        # Apply dropout
        if dropout is not None:
            attention_probs = dropout(attention_probs)

        # Multiply the attention probabilities by the value matrix
        # The output has shape (Batch, h, seq_len, d_k) because we multiply (Batch, h, seq_len, seq_len) by (Batch, h, seq_len, d_k)
        # We also return the attention probabilities to be able to visualize them later on for experiment purposes
        return torch.matmul(attention_probs, value), attention_probs
    
# AddNorm: Add & Norm layer
# We add the 
# We then normalize the result
class AddNorm(nn.Module):
    def __init__(self, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = LayerNorm()

    def forward(self, x, sublayer):
        # return the x and the ouput of the sublayer (e.g. multi-head attention or feed forward)
        # We first apply the normalization to the input x, and then apply the sublayer before adding the result to the input x (residual connection) 
        # Note this is different from the paper where the normalization is applied after the residual connection
        return x + self.dropout(sublayer(self.layer_norm(x)))
    

# Encoder layer: 
# The encoder layer consists of two sublayers: multi-head attention and feed forward
# The input is first passed through a multi-head attention layer
# The result is then passed through a feed forward layer
class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block: MutliHeadAttention, feed_forward_block: FeedForward, dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.skip_connections = nn.ModuleList([AddNorm(dropout) for _ in range(2)])
        
    def forward(self, x, src_mask):
        # Source mask is used to prevent the encoder from paying attention to the padding tokens
        # The first skip connection is applied to the input x and the output of the multi-head attention layer
        x = self.skip_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        # The second skip connection is applied to the output of the first skip connection and the output of the feed forward layer
        x = self.skip_connections[1](x, self.feed_forward_block)

        return x

# Encoder: We stack N encoder layers on top of each other
class Encoder(nn.Module):
    def __init__(self, n_layers: nn.ModuleList):
        super().__init__()
        self.n_layers = n_layers
        self.normalize = LayerNorm()

    def forward(self, x, src_mask):
        # The input is passed through the encoder layers
        for layer in self.n_layers:
            x = layer(x, src_mask)

        # The output of the encoder is normalized
        return self.normalize(x)


# Decoder layer: 
# The decoder layer consists of three sublayers: multi-head attention, cross attention and feed forward
# The input is first passed through a multi-head attention layer
# The result is then passed through a cross attention layer
# The result is then passed through a feed forward layer
class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block: MutliHeadAttention, cross_attention_block: MutliHeadAttention, feed_forward_block: FeedForward, dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.skip_connections = nn.ModuleList([AddNorm(dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, trg_mask):
        # Source mask: used to prevent the encoder from paying attention to the padding tokens
        # Target mask: used to prevent the decoder from paying attention to future words in the target sentence 
        # (e.g. if we are translating "I am a student" to "Je suis étudiant", we don't want the decoder to pay attention to "étudiant" when translating "Je")

        # The first skip connection is applied to the input x and the output of the multi-head attention layer
        x = self.skip_connections[0](x, lambda x: self.self_attention_block(x, x, x, trg_mask))

        # The second skip connection is applied to the output of the first skip connection and the output of the cross attention layer
        x = self.skip_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))

        # The third skip connection is applied to the output of the second skip connection and the output of the feed forward layer
        x = self.skip_connections[2](x, self.feed_forward_block)

        return x
    
class Decoder(nn.Module):
    def __init__(self, n_layers: nn.ModuleList):
        super().__init__()
        self.n_layers = n_layers
        self.normalize = LayerNorm()
    
    def forward(self, x, encoder_output, src_mask, trg_mask):
        # The input is passed through the decoder layers
        for layer in self.n_layers:
            x = layer(x, encoder_output, src_mask, trg_mask)

        # The output of the decoder is normalized
        return self.normalize(x)

# Linear layer: a simple linear transformation
# We map the output of the decoder to the size of the target vocabulary
class LinearLayer(nn.Module):
    def __init__(self, d_model:int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # The input has shape (Batch, seq_len, d_model)
        # The output has shape (Batch, seq_len, vocab_size)
        # We apply the softmax to the last dimension to get a probability distribution over the target vocabulary
        # Note we apply the log softmax because it is more numerically stable
        return torch.log_softmax(self.proj(x), dim=-1) 

# Transformer model:
# The transformer model consists of an encoder, a decoder, input embeddings, positional encodings and a linear layer
# The encoder and decoder are stacked on top of each other
# The input embeddings and positional encodings are applied to the input sentences
# The linear layer is applied to the output of the decoder
class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embedding: InputEmbeddings, trg_embedding: InputEmbeddings, src_position: PositionalEncoding, trg_position: PositionalEncoding, linear_layer: LinearLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embedding = src_embedding
        self.trg_embedding = trg_embedding
        self.src_position = src_position
        self.trg_position = trg_position
        self.linear_layer = linear_layer
    
    def encode(self, src, src_mask):
        # Embed the source sentence and add the positional encoding
        x = self.src_position(self.src_embedding(src))
        # Pass the encoded source sentence through the encoder
        return self.encoder(x, src_mask)
    
    def decode(self, trg, encoder_output, src_mask, trg_mask):
        # Embed the target sentence and add the positional encoding
        x = self.trg_position(self.trg_embedding(trg))
        # Pass the encoded target sentence through the decoder
        return self.decoder(x, encoder_output, src_mask, trg_mask)

    def output(self, x):
        # Apply the linear layer to the output of the decoder
        return self.linear_layer(x)
    
# Create the transformer
def build_transformer(src_vocab_size: int, trg_vocab_size: int, src_seq_len: int, trg_seq_len: int, d_model: int = 512, N: int = 6, h: int = 8, dropout: float = 0.1, d_ff = 2048) -> Transformer:
    # Create the input embeddings
    src_embedding = InputEmbeddings(d_model, src_vocab_size)
    trg_embedding = InputEmbeddings(d_model, trg_vocab_size)
    
    # Create the positional encodings
    src_position = PositionalEncoding(d_model, src_seq_len, dropout)
    trg_position = PositionalEncoding(d_model, trg_seq_len, dropout)

    # Create the encoder layers
    encoder_layers = []
    for _ in range(N):
        self_attention_block = MutliHeadAttention(d_model, h, dropout)
        cross_attention_block = MutliHeadAttention(d_model, h, dropout)
        feed_forward_block = FeedForward(d_model, d_ff, dropout)
        encoder_layers.append(EncoderBlock(self_attention_block, feed_forward_block, dropout))

    # Create the decoder layers
    decoder_layers = []
    for _ in range(N):
        self_attention_block = MutliHeadAttention(d_model, h, dropout)
        cross_attention_block = MutliHeadAttention(d_model, h, dropout)
        feed_forward_block = FeedForward(d_model, d_ff, dropout)
        decoder_layers.append(DecoderBlock(self_attention_block, cross_attention_block, feed_forward_block, dropout))

    # Create the encoder and decoder
    encoder = Encoder(nn.ModuleList(encoder_layers))
    decoder = Decoder(nn.ModuleList(decoder_layers))

    # Create the linear layer
    linear_layer = LinearLayer(d_model, trg_vocab_size)

    # Create the transformer
    transformer = Transformer(encoder, decoder, src_embedding, trg_embedding, src_position, trg_position, linear_layer)

    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer