import pytest
import torch
import sys
sys.path.append('..')

from src.data.tokenizer import SimpleTokenizer

class TestTokenizer:
    def setup_method(self):
        self.tokenizer = SimpleTokenizer(max_vocab_size=100)
        self.texts = ["def hello world", "return max value", "print('hello')"]
    
    def test_build_vocab(self):
        self.tokenizer.build_vocab(self.texts)
        assert len(self.tokenizer.word2idx) <= 100
        assert '<pad>' in self.tokenizer.word2idx
        assert '<sos>' in self.tokenizer.word2idx
    
    def test_encode_decode(self):
        self.tokenizer.build_vocab(self.texts)
        text = "hello world"
        encoded = self.tokenizer.encode(text, max_len=10)
        decoded = self.tokenizer.decode(encoded)
        assert isinstance(encoded, torch.Tensor)
        assert encoded.shape[0] == 10