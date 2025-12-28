import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import json
from pathlib import Path


class CharacterVocabulary:
    """manages character-to-index mappings."""
    
    def __init__(self, char_vocab_path: str = 'data/char_vocab.json'):
        with open(char_vocab_path, 'r', encoding='utf-8') as f:
            chars = json.load(f)
        
        # special tokens
        self.PAD_TOKEN = '<PAD>'
        self.SOS_TOKEN = '<SOS>'
        self.EOS_TOKEN = '<EOS>'
        self.UNK_TOKEN = '<UNK>'
        
        self.special_tokens = [self.PAD_TOKEN, self.SOS_TOKEN, 
                              self.EOS_TOKEN, self.UNK_TOKEN]
        
        # build vocabulary
        self.char2idx = {token: idx for idx, token in enumerate(self.special_tokens)}
        for char in chars:
            if char not in self.char2idx:
                self.char2idx[char] = len(self.char2idx)
        
        self.idx2char = {idx: char for char, idx in self.char2idx.items()}
        
        self.pad_idx = self.char2idx[self.PAD_TOKEN]
        self.sos_idx = self.char2idx[self.SOS_TOKEN]
        self.eos_idx = self.char2idx[self.EOS_TOKEN]
        self.unk_idx = self.char2idx[self.UNK_TOKEN]
    
    def encode(self, word: str, add_eos: bool = False) -> list:
        """convert word to list of indices."""
        indices = [self.char2idx.get(c, self.unk_idx) for c in word]
        if add_eos:
            indices.append(self.eos_idx)
        return indices
    
    def decode(self, indices: list, stop_at_eos: bool = True) -> str:
        """convert list of indices to word."""
        chars = []
        for idx in indices:
            if stop_at_eos and idx == self.eos_idx:
                break
            if idx in [self.pad_idx, self.sos_idx, self.eos_idx]:
                continue
            chars.append(self.idx2char.get(idx, self.UNK_TOKEN))
        return ''.join(chars)
    
    def __len__(self):
        return len(self.char2idx)


class Encoder(nn.Module):
    """bidirectional LSTM encoder."""
    
    def __init__(self, vocab_size: int, embedding_dim: int, 
                 hidden_dim: int, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc_hidden = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc_cell = nn.Linear(hidden_dim * 2, hidden_dim)
    
    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> Tuple:
        embedded = self.embedding(x)
        
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        outputs, (hidden, cell) = self.lstm(packed)
        
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        
        hidden = hidden.view(self.num_layers, 2, -1, self.hidden_dim)
        cell = cell.view(self.num_layers, 2, -1, self.hidden_dim)
        
        hidden = torch.cat([hidden[:, 0, :, :], hidden[:, 1, :, :]], dim=2)
        cell = torch.cat([cell[:, 0, :, :], cell[:, 1, :, :]], dim=2)
        
        hidden = torch.tanh(self.fc_hidden(hidden))
        cell = torch.tanh(self.fc_cell(cell))
        
        return outputs, hidden, cell


class Attention(nn.Module):
    """Bahdanau attention mechanism."""
    
    def __init__(self, encoder_dim: int, decoder_dim: int, attention_dim: int):
        super().__init__()
        
        self.encoder_proj = nn.Linear(encoder_dim, attention_dim)
        self.decoder_proj = nn.Linear(decoder_dim, attention_dim)
        self.v = nn.Linear(attention_dim, 1, bias=False)
    
    def forward(self, decoder_hidden: torch.Tensor, 
                encoder_outputs: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple:
        
        decoder_proj = self.decoder_proj(decoder_hidden)
        encoder_proj = self.encoder_proj(encoder_outputs)
        
        combined = torch.tanh(encoder_proj + decoder_proj.unsqueeze(1))
        
        energy = self.v(combined).squeeze(2)
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        
        attention_weights = F.softmax(energy, dim=1)
        
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)
        context = context.squeeze(1)
        
        return context, attention_weights


class Decoder(nn.Module):
    """LSTM decoder with attention."""
    
    def __init__(self, vocab_size: int, embedding_dim: int,
                 encoder_dim: int, decoder_dim: int, attention_dim: int,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.decoder_dim = decoder_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)
        
        self.lstm = nn.LSTM(
            embedding_dim + encoder_dim,
            decoder_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc_out = nn.Linear(embedding_dim + encoder_dim + decoder_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input_char: torch.Tensor,
                hidden: torch.Tensor, cell: torch.Tensor,
                encoder_outputs: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple:
        
        embedded = self.dropout(self.embedding(input_char))
        
        context, attention_weights = self.attention(
            hidden[-1], encoder_outputs, mask
        )
        
        lstm_input = torch.cat([embedded, context.unsqueeze(1)], dim=2)
        
        lstm_output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        
        output_input = torch.cat([
            embedded.squeeze(1),
            context,
            lstm_output.squeeze(1)
        ], dim=1)
        
        output = self.fc_out(output_input)
        
        return output, hidden, cell, attention_weights


class Seq2SeqSpellchecker(nn.Module):
    """Seq2Seq model for Georgian spelling correction."""
    
    def __init__(self, vocab_size: int,
                 embedding_dim: int = 128,
                 encoder_dim: int = 256,
                 decoder_dim: int = 256,
                 attention_dim: int = 128,
                 num_layers: int = 2,
                 dropout: float = 0.3):
        super().__init__()
        
        self.encoder = Encoder(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=encoder_dim,
            num_layers=num_layers,
            dropout=dropout
        )
        
        self.decoder = Decoder(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            encoder_dim=encoder_dim * 2,
            decoder_dim=decoder_dim,
            attention_dim=attention_dim,
            num_layers=num_layers,
            dropout=dropout
        )
    
    def forward(self, src: torch.Tensor, src_lengths: torch.Tensor,
                trg: torch.Tensor, teacher_forcing_ratio: float = 0.5) -> torch.Tensor:
        """training forward pass with teacher forcing."""
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        vocab_size = self.decoder.vocab_size
        
        encoder_outputs, hidden, cell = self.encoder(src, src_lengths)
        
        # create mask matching encoder_outputs shape
        # encoder_outputs: [batch, seq_len, hidden]
        max_len = encoder_outputs.shape[1]
        mask = torch.arange(max_len, device=src.device).unsqueeze(0).expand(batch_size, -1)
        mask = (mask < src_lengths.unsqueeze(1)).long()
        
        outputs = torch.zeros(batch_size, trg_len, vocab_size).to(src.device)
        
        decoder_input = trg[:, 0].unsqueeze(1)
        
        for t in range(1, trg_len):
            output, hidden, cell, _ = self.decoder(
                decoder_input, hidden, cell, encoder_outputs, mask
            )
            
            outputs[:, t, :] = output
            
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            
            if teacher_force:
                decoder_input = trg[:, t].unsqueeze(1)
            else:
                decoder_input = output.argmax(1).unsqueeze(1)
        
        return outputs
    
    def predict(self, src: torch.Tensor, src_lengths: torch.Tensor,
                max_length: int = 30, sos_idx: int = 1, eos_idx: int = 2) -> torch.Tensor:
        """inference: generate corrected word."""
        batch_size = src.shape[0]
        device = src.device
        
        encoder_outputs, hidden, cell = self.encoder(src, src_lengths)
        
        # create mask matching encoder_outputs shape
        max_len = encoder_outputs.shape[1]
        mask = torch.arange(max_len, device=device).unsqueeze(0).expand(batch_size, -1)
        mask = (mask < src_lengths.unsqueeze(1)).long()
        
        decoder_input = torch.full((batch_size, 1), sos_idx, 
                                  dtype=torch.long, device=device)
        
        predictions = torch.zeros(batch_size, max_length, 
                                 dtype=torch.long, device=device)
        predictions[:, 0] = sos_idx
        
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        for t in range(1, max_length):
            output, hidden, cell, _ = self.decoder(
                decoder_input, hidden, cell, encoder_outputs, mask
            )
            
            predicted_char = output.argmax(1)
            
            predictions[:, t] = torch.where(finished, 
                                           torch.zeros_like(predicted_char),
                                           predicted_char)
            
            finished = finished | (predicted_char == eos_idx)
            
            if finished.all():
                break
            
            decoder_input = predicted_char.unsqueeze(1)
        
        return predictions


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)