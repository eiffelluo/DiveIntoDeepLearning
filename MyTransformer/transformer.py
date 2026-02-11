import torch
import torch.nn as nn
import dataloader
import pos_encoder
import mulhead_attention
import add_norm
import position_wise_ffn

class Encoder(nn.Module):
    """The base encoder interface for the encoder--decoder architecture.

    Defined in :numref:`sec_encoder-decoder`"""
    def __init__(self):
        super().__init__()

    # Later there can be additional arguments (e.g., length excluding padding)
    def forward(self, X, *args):
        raise NotImplementedError

class Decoder(nn.Module):
    """The base decoder interface for the encoder--decoder architecture.

    Defined in :numref:`sec_encoder-decoder`"""
    def __init__(self):
        super().__init__()

    # Later there can be additional arguments (e.g., length excluding padding)
    def init_state(self, enc_all_outputs, *args):
        raise NotImplementedError

    def forward(self, X, state):
        raise NotImplementedError
    
class AttentionDecoder(Decoder):
    """The base attention-based decoder interface.

    Defined in :numref:`sec_seq2seq_attention`"""
    def __init__(self):
        super().__init__()

    @property
    def attention_weights(self):
        raise NotImplementedError

class EncoderDecoder(nn.Module):
    def __init__(self):
        super().__init__()


class EncodeBlock(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                ffn_num_input, ffn_num_hiddens, num_heads,
                dropout, use_bias=False, **kwargs):
        super().__init__(**kwargs)
        
        # self.mAttention = mulhead_attention.MultiHeadAttention(query_size, key_size,value_size, num_hiddens,num_heads)
        self.mAttention = mulhead_attention.DotProductAttention(0)
        self.addition = add_norm.Add()
        self.norm = add_norm.FeatureNorm()
        self.ffn = position_wise_ffn.PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addition2 = add_norm.Add(dropout)
        self.norm2 = add_norm.FeatureNorm()

    def forward(self,X,X_valid_len):
        Y = self.norm(self.addition(X,self.mAttention(X,X,X,X_valid_len)))
        return self.norm2(self.addition2(Y,self.ffn(Y)))
        
class TransformerEncoder(Encoder):
    def __init__(self, vocab_size, key_size, query_size, value_size,
                 num_hiddens, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout, use_bias=False, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.num_layers = num_layers
        # 创建 Embedding 层
        # num_embeddings: 词汇表大小
        # embedding_dim: 嵌入维度
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=num_hiddens)
        self.posEncoder = pos_encoder.PosEncoder(num_hiddens,dropout)
        self.blks = nn.Sequential()

        for i in range(num_layers):
            self.blks.add_module('block'+str(i),EncodeBlock(key_size, query_size, value_size,num_hiddens,
            ffn_num_input, ffn_num_hiddens,num_heads, dropout, use_bias))
        

    def forward(self,X,X_valid_len):
        X = self.posEncoder(self.embedding(X))
        for i, blk in enumerate(self.blks):
            X = blk(X,X_valid_len)
        return X
        
        
class DecoderBlock(nn.Module):
    """解码器中第i个块"""
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout, i, **kwargs):
        super(DecoderBlock, self).__init__(**kwargs)

        self.mAttention = mulhead_attention.MultiHeadAttention(query_size, key_size,value_size, num_hiddens,num_heads)
        self.addition = add_norm.Add()
        self.norm = add_norm.FeatureNorm()
        self.ffn = position_wise_ffn.PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addition2 = add_norm.Add(dropout)
        self.norm2 = add_norm.FeatureNorm()

    def forward(self,X,X_valid_len):
        Y = self.norm(self.addition(X,self.mAttention(X,X,X,X_valid_len)))
        return self.norm2(self.addition2(Y,self.ffn(Y)))


class TransformerDecoder(AttentionDecoder):
    def __init__(self, vocab_size, key_size, query_size, value_size,
                 num_hiddens, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout, **kwargs):
        super(TransformerDecoder, self).__init__(**kwargs)

    def forward(self,X,X_valid_len):
        X = self.posEncoder(self.embedding(X))
        for i, blk in enumerate(self.blks):
            X = blk(X,X_valid_len)
        return X


def main():
    batch_size = 5
    num_step = 10
    data_iter, src_vocab, tgt_vocab = dataloader.load_data_nmt(batch_size,num_step)
    vocab_size = len(src_vocab)
    num_hiddens = 4
    ffn_num_hiddens = 50
    num_heads = 2
    num_layers = 2
    dropout = 0
    use_bias = False

    encoder = TransformerEncoder(vocab_size=vocab_size, 
                                key_size=num_hiddens, 
                                query_size=num_hiddens,
                                value_size=num_hiddens,
                                num_hiddens=num_hiddens, 
                                ffn_num_input=num_hiddens, 
                                ffn_num_hiddens=ffn_num_hiddens,
                                num_heads=num_heads, 
                                num_layers=num_layers, 
                                dropout=dropout, 
                                use_bias=use_bias)
    for batch in data_iter:
        X, X_valid_len, Y, Y_valid_len = batch
        output = encoder(X,X_valid_len)
        print(output)


main()