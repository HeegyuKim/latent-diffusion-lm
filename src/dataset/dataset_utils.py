from typing import List, Any, Optional



def normalize_weights(weights):
    if weights is None:
        return None
        
    s = sum(weights)
    return [w / s for w in weights]
    
def pad_to_multiple_of(pad_size: int, multiple_of: int):
    if pad_size % multiple_of == 0:
        return pad_size
    else:
        return multiple_of * ((pad_size // multiple_of) + 1)

def truncate(
    arr: List, 
    max_len: int, 
    truncation_side: str = "right", 
    prefix_value: Optional[Any] = None,
    postfix_value: Optional[Any] = None
    ) -> List:
    if len(arr) > max_len:
        if truncation_side == "right":
            arr = arr[:max_len]
        else:
            arr = arr[-max_len:]

    if prefix_value is not None:
        arr.insert(0, prefix_value)
    if postfix_value is not None:
        arr.append(postfix_value)

    return arr

def pad(
    arr: List, 
    max_len: int, 
    padding_side: str = "right", 
    padding_value: Any = 0
    ) -> List:
    if len(arr) < max_len:
        p = [padding_value] * (max_len - len(arr))
        
        if padding_side == "right":
            arr = arr + p
        else:
            arr = p + arr
    
    return arr

def get_longest_length(arr):
    return max(map(len, arr))