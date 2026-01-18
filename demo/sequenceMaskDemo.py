import torch

# a = torch.tensor([1, 2, 3])   # shape: (3,)
# print(a.shape)

# b = a[:, None]                    # shape: (3, 1)
# print(b)
# print(b.shape)


def sequence_mask(X, valid_len, value=0):
    """在序列中屏蔽不相关的项"""
    maxlen = X.size(1)
    a = torch.arange((maxlen), dtype=torch.float32,
                     device=X.device)
    a2 = a[None, :]
    new_valid_len = valid_len[:, None]
    mask = a2 < new_valid_len
    print(mask)
    X[~mask] = value
    return X

# X = torch.tensor([[1, 2, 3], [4, 5, 6]])
# print(sequence_mask(X, torch.tensor([1, 2])))

X = torch.ones(2, 3, 4)
print(sequence_mask(X, torch.tensor([1, 2]), value=-1))


# valid_len = torch.tensor([1, 2])
# new_valid_len = valid_len[:, None]
# print(new_valid_len)
# print(new_valid_len.shape)