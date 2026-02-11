import torch

def read_data_nmt():
    with open('../data/fra-eng/fra.txt', 'r', encoding='utf-8') as f:
    # with open('../data/fra-eng/fra_tiny.txt', 'r', encoding='utf-8') as f:
        return f.read()

def preprocess_nmt(text):
    def no_space(char, prev_char):
        return char in set(',.!?') and prev_char != ' '

    text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()
    out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char for i, char in enumerate(text)]
    return ''.join(out)

def tokenize_nmt(text, num_examples=None):
    source, target = [], []
    for i, line in enumerate(text.split('\n')):
        if num_examples and i >= num_examples:
            break
        parts = line.split('\t')
        if len(parts) == 2:
            source.append(parts[0].split(' '))
            target.append(parts[1].split(' '))
    return source, target

class Vocab:  #@save
    """文本词表"""
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        self._idx2tokens = []
        self._tokens2idx = {}
        self._tokens2freq = {}

        token2freqDict = {}

        for line in tokens:
            for token in line: 
                if token not in token2freqDict:
                    token2freqDict[token] = 0
                
                token2freqDict[token] += 1

        for token, freq in token2freqDict.items():
            if freq >= min_freq:
                self._tokens2freq[token] = freq

        if(reserved_tokens == None):
            reserved_tokens = []
        self._idx2tokens =  ['<unk>'] + reserved_tokens + sorted(self._tokens2freq, key=self._tokens2freq.get,reverse=True)
      
        for i in range(len(self._idx2tokens)):
            self._tokens2idx[self._idx2tokens[i]] = i

        # print(self._tokens2freq)
        # print(self._idx2tokens)
        # print(self._tokens2idx)

    def __len__(self):
        return len(self._idx2tokens)


    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self._tokens2idx.get(tokens,self.unk)
            
        return [self.__getitem__(token) for token in tokens]
    
    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self._idx2tokens[indices]
        return [self.to_tokens(idx) for idx in indices ]
    
    @property
    def unk(self):
        return 0

def truncate_pad(token_array,num_steps: int, pad_token:int):
    padding_num  = num_steps - len(token_array)
    padding_array = token_array +  [ pad_token ] * padding_num  if  padding_num > 0  else token_array[:num_steps]
    return padding_array

def build_array_nmt(lines, vocab, num_steps):
    lines = [vocab[l]   for l in lines]
    lines = [l + [vocab['<eos>']] for l in lines]
    array = torch.tensor([truncate_pad(l, num_steps, vocab['<pad>']) for l in lines])
    valid_len = (array != vocab['<pad>']).type(torch.int32).sum(1)
    return array, valid_len

class DataIterator:
    """迭代器 - 实现 __iter__ 和 __next__"""
    def __init__(self, data_arrays,batch_size):
        
        self.src_array = data_arrays[0]
        self.src_valid_len = data_arrays[1]
        self.tgt_array = data_arrays[2]
        self.tgt_valid_len = data_arrays[3]

        self.index = 0
        self.batch_size = batch_size
    
    def __iter__(self):
        print("MyIterator.__iter__ 被调用：返回自身")
        return self
    
    def __next__(self):
        if self.index >= len(self.src_array):
            raise StopIteration
        
        # indices = range(self.index,self.index+self.batch_size)
        indices = [(self.index + i) % len(self.src_array) for i in range(self.batch_size)]
        src_batch = self.src_array[indices]
        src_valid_len_batch = self.src_valid_len[indices]
        tgt_batch = self.tgt_array[indices]
        tgt_valid_len_batch = self.tgt_valid_len[indices]
        self.index += self.batch_size
        return (src_batch,src_valid_len_batch,tgt_batch,tgt_valid_len_batch)

def load_array(data_arrays,batch_size):
    iter = DataIterator(data_arrays,batch_size)
    return iter

      
def load_data_nmt(batch_size, num_steps, num_examples=600):
    text = preprocess_nmt(read_data_nmt())
    source, target = tokenize_nmt(text, num_examples)
    src_vocab = Vocab(source, min_freq=1, reserved_tokens=['<pad>', '<bos>', '<eos>'])
    tgt_vocab = Vocab(target, min_freq=1, reserved_tokens=['<pad>', '<bos>', '<eos>'])
    src_array, src_valid_len = build_array_nmt(source, src_vocab, num_steps)
    # print(src_array)
    # print(src_valid_len)
    tgt_array, tgt_valid_len = build_array_nmt(target, tgt_vocab, num_steps)
    data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)
    data_iter = load_array(data_arrays, batch_size)
    return data_iter, src_vocab, tgt_vocab

def main():
    # text = read_data_nmt()
    # print(text[:100])

    # text = preprocess_nmt(text)
    # print(text[:100])

    # source = ['ab','ac','ab','a','ab','ac']
    # src_vocab = Vocab(source,min_freq=2,reserved_tokens=['<bos>','<eos>'])
    # print(src_vocab['ab'])
    # print(src_vocab.get_token(0))
    # src_array, src_valid_len = build_array_nmt(source, src_vocab, 10)
    # print(src_array)
    # print(src_valid_len)

    data_iter, src_vocab, tgt_vocab = load_data_nmt(5,10)
    for batch in data_iter:
        X, X_valid_len, Y, Y_valid_len = batch
        # print(X)
        # print(X_valid_len)
        # print(Y)
        # print(Y_valid_len)
       


# main()