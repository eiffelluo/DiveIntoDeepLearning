import torch
import torch.nn as nn
import torch.optim as optim
from d2l import torch as d2l
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

class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    """带遮蔽的softmax交叉熵损失函数"""
    # pred的形状：(batch_size,num_steps,vocab_size)
    # label的形状：(batch_size,num_steps)
    # valid_len的形状：(batch_size,)
    def forward(self, pred, label, valid_len):
        weight = torch.ones_like(label)
        masked_weight = mulhead_attention.sequence_mask(weight, valid_len)
        # print(masked_weight)
        self.reduction='none'
        unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(pred.permute(0, 2, 1), label)
        weighted_loss = (unweighted_loss * masked_weight).mean(dim=1)
        return weighted_loss
        
        

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
    def __init__(self,encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self,X,X_valid_len, Y, Y_valid_len):
        enc_output = self.encoder(X,X_valid_len)
        # print(enc_output)
        output2 = self.decoder(Y,Y_valid_len,(enc_output,X_valid_len))
        # print(output2)
        return output2



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
        #自注意力
        self.decoderAttention = mulhead_attention.MultiHeadAttention(query_size, key_size,value_size, num_hiddens,num_heads)
        self.decoderAddition = add_norm.Add()
        self.decoderNorm = add_norm.FeatureNorm()
        
        # 下面这段逻辑和编码器一样
        self.mAttention = mulhead_attention.MultiHeadAttention(query_size, key_size,value_size, num_hiddens,num_heads)
        self.addition = add_norm.Add()
        self.norm = add_norm.FeatureNorm()
        self.ffn = position_wise_ffn.PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addition2 = add_norm.Add(dropout)
        self.norm2 = add_norm.FeatureNorm()
      

    def forward(self,X,X_valid_len,state):
        enc_outputs, enc_valid_lens = state[0], state[1]
        num_step = X.shape[1]
        # 掩蔽一行当前token后面的token
        valid_len = create_sequence_matrix_vectorized(X_valid_len, num_step,X.device)
        # valid_len2 = create_dec_valid_lens(X.shape[0],num_step,X.device)
        Y = self.decoderNorm(self.decoderAddition(X,self.decoderAttention(X,X,X,valid_len)))
      
        Y2 = self.norm(self.addition(Y,self.mAttention(Y,enc_outputs,enc_outputs,enc_valid_lens)))
        return self.norm2(self.addition2(Y2,self.ffn(Y2)))
    

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

        self.mlp = nn.Linear(num_hiddens,vocab_size)

        for i in range(num_layers):
            self.blks.add_module('block'+str(i),DecoderBlock(key_size, query_size, value_size,num_hiddens,
            ffn_num_input, ffn_num_hiddens,num_heads, dropout,i))

    @property
    def attention_weights(self):
        raise NotImplementedError
    

    def forward(self,X,X_valid_len,state):
        X = self.posEncoder(self.embedding(X))
        for i, blk in enumerate(self.blks):
            X = blk(X,X_valid_len,state)
        
        return self.mlp(X)

def train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device):
    optimizer = optim.SGD(net.parameters(), lr=lr)  # 优化器
    for epoch in range(num_epochs):
        loss = MaskedSoftmaxCELoss()
        for batch in train_iter:
            X, X_valid_len, Y, Y_valid_len = batch
            Y_hat = net(X, X_valid_len, Y, Y_valid_len)
            l = loss(Y_hat,Y,Y_valid_len)
            print(l)

            # 反向传播
            optimizer.zero_grad()  # 清空梯度
            l.sum().backward()        # 计算梯度
            optimizer.step()      # 更新参数

def predict_seq2seq(net, src_sentence, src_vocab, tgt_vocab, num_steps,
                    device, save_attention_weights=False):  
    pass

def main():
    device = d2l.try_gpu()
    num_epochs = 10
    lr = 0.01
    batch_size = 5
    num_step = 10
    train_iter, src_vocab, tgt_vocab = dataloader.load_data_nmt(batch_size,num_step)
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
    
    net = EncoderDecoder(encoder, decoder)

    train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)



main()