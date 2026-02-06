import torch
import torch.nn as nn
import dataloader
import pos_encoder

class EncoderDecoder(nn.Module):
    def __init__(self):
        super().__init__()


class EncodeBlock(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self,X,X_valid_len):
        pass


class TransformerEncoder(nn.Module):
    def __init__(self,num_step,num_embeddings,embedding_dim,blk_num,**kwargs):
        super().__init__()
        # 创建 Embedding 层
        # num_embeddings: 词汇表大小
        # embedding_dim: 嵌入维度
        self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        self.posEncoder = pos_encoder.PosEncoder(embedding_dim,num_step)
        self.blks = []
        

    def forward(self,X,X_valid_len):
        embedX = self.embedding(X)
        posX = self.posEncoder(X)
        



class TransformerDecoder(nn.Module):
    def __init__(self):
        super().__init__()


def main():
    data_iter, src_vocab, tgt_vocab = dataloader.load_data_nmt(5,10)
    src_num_embeddings = len(src_vocab)
    embedding_dim = 300
    encoder = TransformerEncoder(src_num_embeddings,embedding_dim)
    for batch in data_iter:
        X, X_valid_len, Y, Y_valid_len = batch
        Y_hat,state = encoder(X,X_valid_len)


main()