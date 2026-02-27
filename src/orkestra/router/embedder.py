"""Longformer embeddings for query classification."""

import logging
import os
import warnings

import numpy as np

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import torch
from transformers import AutoModel, AutoTokenizer
from transformers import logging as transformers_logging

transformers_logging.set_verbosity_error()
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

_MODEL_NAME = "allenai/longformer-base-4096"
EMBEDDING_DIM = 768

_tokenizer = None
_model = None
_device = None


def _get_device() -> torch.device:
    global _device
    if _device is not None:
        return _device
    if torch.cuda.is_available():
        _device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        _device = torch.device("mps")
    else:
        _device = torch.device("cpu")
    return _device


def _init_model():
    global _tokenizer, _model
    if _tokenizer is not None and _model is not None:
        return
    device = _get_device()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
        _model = AutoModel.from_pretrained(_MODEL_NAME).to(device)
    _model.eval()


def get_longformer_embedding(text: str) -> np.ndarray:
    """Compute a 768-dim Longformer embedding for a query string."""
    _init_model()
    device = _get_device()

    inputs = _tokenizer( #type:ignore
        [text],
        padding=True,
        truncation=True,
        max_length=4096,
        return_tensors="pt",
    ).to(device)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*automatically padded.*")
        with torch.no_grad():
            outputs = _model(**inputs) #type:ignore
            last_hidden_state = outputs.last_hidden_state

    attention_mask = inputs["attention_mask"]
    mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    embedding = (last_hidden_state * mask_expanded).sum(1) / mask_expanded.sum(1)

    return embedding[0].cpu().numpy()
