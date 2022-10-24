import torch
from typing import List


def create_bow(arr: List[List[int]], vocab_size: int, skip_tokens: List[int] = []):
    x = torch.zeros((len(arr), vocab_size))
    for i, v in enumerate(arr):
        for t in v:
            if t not in skip_tokens:
                x[i, t] = 1 # += 1 대신 +1
    return x


# def create_bow(arr: List[List[int]], vocab_size: int, skip_tokens: List[int] = []):
#     x = torch.zeros((len(arr), vocab_size))
#     for i, v in enumerate(arr):
#         for t in v:
#             if t not in skip_tokens:
#                 x[i, t] += 1
#     return x


def discrete_kl_div(inputs, targets):
    # return (targets * (inputs / targets)).mean()
    return -targets * (inputs / targets).log()

if __name__ == "__main__":
    # bow1 = create_bow([[0,1,2,3,4,1,2,3,4],[0,0,0,0,1,2,2,2,2,3,4,5,6,7]], 10, [0])
    bow2 = create_bow([[0,1,2,3,5,6,7,8,6,4,6,2,9],[0,0,0,2,1,3,3,3,3,3,3,1,1,1,2,2,2,3,4,5,6,7]], 10, [0])
    bow1 = create_bow([[0,1,2,3,5,6,7,8,6,4,6,2],[0,0,0,2,1,3,3,3,3,3,3,1,1,1,2,2,2,3,4,5,6,7]], 10, [0])
    print(bow1, bow2)
    from torch.nn import functional as F

    # print(discrete_kl_div(bow1, bow2, reduction="batchmean"))
    # print(discrete_kl_div(bow1, bow2))
    print(F.binary_cross_entropy(bow1, bow2, reduction="mean"))
