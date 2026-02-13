import pytest
import torch
import sys
sys.path.append('..')

from src.models.encoder import EncoderRNN
from src.models.vanilla_rnn import VanillaDecoder, VanillaSeq2Seq
from src.models.lstm import LSTMDecoder, LSTMSeq2Seq
from src.models.attention import AttentionDecoder, AttentionSeq2Seq

class TestModels:
    def setup_method(self):
        self.batch_size = 2
        self.src_len = 10
        self.tgt_len = 15
        self.src_vocab = 100
        self.tgt_vocab = 100
        self.emb_dim = 32
        self.hidden_dim = 64
        self.device = torch.device('cpu')
        
        self.src = torch.randint(0, self.src_vocab, (self.batch_size, self.src_len))
        self.tgt = torch.randint(0, self.tgt_vocab, (self.batch_size, self.tgt_len))
    
    def test_vanilla_rnn(self):
        encoder = EncoderRNN(self.src_vocab, self.emb_dim, self.hidden_dim, rnn_type='rnn')
        decoder = VanillaDecoder(self.tgt_vocab, self.emb_dim, self.hidden_dim)
        model = VanillaSeq2Seq(encoder, decoder, self.device)
        
        output = model(self.src, self.tgt)
        assert output.shape == (self.batch_size, self.tgt_len-1, self.tgt_vocab)
    
    def test_lstm(self):
        encoder = EncoderRNN(self.src_vocab, self.emb_dim, self.hidden_dim, rnn_type='lstm')
        decoder = LSTMDecoder(self.tgt_vocab, self.emb_dim, self.hidden_dim)
        model = LSTMSeq2Seq(encoder, decoder, self.device)
        
        output = model(self.src, self.tgt)
        assert output.shape == (self.batch_size, self.tgt_len-1, self.tgt_vocab)
    
    def test_attention(self):
        encoder = EncoderRNN(self.src_vocab, self.emb_dim, self.hidden_dim, rnn_type='bidirectional_lstm')
        decoder = AttentionDecoder(self.tgt_vocab, self.emb_dim, self.hidden_dim*2, self.hidden_dim)
        model = AttentionSeq2Seq(encoder, decoder, self.device)
        
        output, attentions = model(self.src, self.tgt, return_attention=True)
        assert output.shape == (self.batch_size, self.tgt_len-1, self.tgt_vocab)
        assert attentions.shape == (self.batch_size, self.tgt_len-1, self.src_len)