import nltk
import torch
from nltk.translate.bleu_score import sentence_bleu


def generate_translation(transformer, src_sequence, target_sequence):
    """
    Generate a translation using the provided Transformer model.

    Args:
    - transformer: The Transformer model
    - src_sequence: The source sequence (input sentence)

    Returns:
    - pred_sequence: The predicted translation
    """
    # Convert the source sequence to a PyTorch tensor
    src_tensor = torch.tensor(src_sequence).unsqueeze(0)  # Add batch dimension

    # Perform a forward pass through the Transformer model
    pred_tensor = transformer(src_tensor)

    # Convert the predicted tensor back to a list
    pred_sequence = pred_tensor.squeeze(0).tolist()

    # Compute the BLEU score
    bleu_score = sentence_bleu([target_sequence], pred_sequence)

    return bleu_score, pred_sequence