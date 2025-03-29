import numpy as np

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def find_zero_from_end(arr):
    arr_np = np.array(arr)
    row_sums = np.sum(arr_np, axis=1)

    min_i = len(row_sums)
    for i in range(len(row_sums) - 1, -1, -1):
        if row_sums[i] == 0 and np.all(row_sums[i:] == 0):
            min_i = min(i, min_i)
        else:
            break
    return min_i

def find_zero_from_start(arr):
    arr_np = np.array(arr)
    row_sums = np.sum(arr_np, axis=1)

    max_i = 0
    for i in range(len(row_sums)):
        if row_sums[i] == 0 and np.all(row_sums[:i] == 0):
            max_i = max(i + 1, max_i)
        else:
            break
    return max_i