import torch
import torch.nn as nn
import dataloader
import pos_encoder
import mulhead_attention
import add_norm
import position_wise_ffn

def create_sequence_matrix_vectorized(X, n,device):
    """
    向量化实现的版本，效率更高
    """
    batch_size = len(X)
    col_indices = torch.arange(n,device=device).unsqueeze(0).expand(batch_size, n)
    
    # 创建结果矩阵：列索引+1，但只保留小于对应长度的位置
    result = (col_indices + 1) * (col_indices < X.unsqueeze(1))
    
    return result

def create_dec_valid_lens(batch_size,num_steps,device):
    # dec_valid_lens的开头:(batch_size,num_steps),
    # 其中每一行是[1,2,...,num_steps]
    dec_valid_lens = torch.arange(
        1, num_steps + 1, device=device).repeat(batch_size, 1)
    return dec_valid_lens

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
        
        self.mAttention = mulhead_attention.MultiHeadAttention(query_size, key_size,value_size, num_hiddens,num_heads)
        # self.mAttention = mulhead_attention.MulHeadAttention(num_heads,num_hiddens,query_size, key_size,value_size, num_hiddens)
        # self.mAttention = mulhead_attention.DotProductAttention(0)
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
        num_step = X.shape[1]
        # 掩蔽一行当前token后面的token
        valid_len = create_sequence_matrix_vectorized(X_valid_len, num_step)
        # valid_len2 = create_dec_valid_lens(X.shape[0],num_step,X.device)
        Y = self.norm(self.addition(X,self.mAttention(X,X,X,valid_len)))
        return self.norm2(self.addition2(Y,self.ffn(Y)))


class TransformerDecoder(AttentionDecoder):
    def __init__(self, vocab_size, key_size, query_size, value_size,
                 num_hiddens, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout, **kwargs):
        super(TransformerDecoder, self).__init__(**kwargs)
        self.num_layers = num_layers
        # 创建 Embedding 层
        # num_embeddings: 词汇表大小
        # embedding_dim: 嵌入维度
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=num_hiddens)
        self.posEncoder = pos_encoder.PosEncoder(num_hiddens,dropout)
        self.blks = nn.Sequential()

        for i in range(num_layers):
            self.blks.add_module('block'+str(i),DecoderBlock(key_size, query_size, value_size,num_hiddens,
            ffn_num_input, ffn_num_hiddens,num_heads, dropout,i))

    @property
    def attention_weights(self):
        raise NotImplementedError
    

    def forward(self,X,X_valid_len):
        X = self.posEncoder(self.embedding(X))
        for i, blk in enumerate(self.blks):
            X = blk(X,X_valid_len)
        return X


def main():
    batch_size = 5
    num_step = 10
    data_iter, src_vocab, tgt_vocab = dataloader.load_data_nmt(batch_size,num_step)
    src_vocab_size = len(src_vocab)
    rgt_vocab_size = len(tgt_vocab)
    num_hiddens = 4
    ffn_num_hiddens = 50
    num_heads = 2
    num_layers = 2
    dropout = 0
    use_bias = False

    encoder = TransformerEncoder(vocab_size=src_vocab_size, 
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
    
    decoder = TransformerDecoder(vocab_size=rgt_vocab_size, 
                                key_size=num_hiddens, 
                                query_size=num_hiddens,
                                value_size=num_hiddens,
                                num_hiddens=num_hiddens, 
                                ffn_num_input=num_hiddens, 
                                ffn_num_hiddens=ffn_num_hiddens,
                                num_heads=num_heads, 
                                num_layers=num_layers, 
                                dropout=dropout)
    
    for batch in data_iter:
        X, X_valid_len, Y, Y_valid_len = batch
        # output = encoder(X,X_valid_len)
        # print(output)
        output2 = decoder(Y,Y_valid_len)
        print(output2)



main()