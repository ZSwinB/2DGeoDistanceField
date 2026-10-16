import numpy as np
from numba import njit
@njit(nogil=True, cache=True)
def repair_near_angles(ea, ei, order, cnt):
    lo = 0
    while lo < cnt:
        hi = lo + 1
        while hi < cnt and ea[order[hi]] - ea[order[hi - 1]] < 1e-12:
            hi += 1
        needs_repair = hi - lo > 1 and ea[order[hi - 1]] != ea[order[lo]]
        if hi - lo > 1 and (not needs_repair):
            for q in range(lo + 1, hi):
                if ei[order[q]] < ei[order[q - 1]]:
                    needs_repair = True
                    break
        if needs_repair:
            group = np.sort(order[lo:hi])
            for i in range(1, len(group)):
                key = group[i]
                j = i - 1
                while j >= 0:
                    a1 = ea[group[j]]
                    a2 = ea[key]
                    if a1 > a2:
                        group[j + 1] = group[j]
                    elif abs(a1 - a2) < 1e-12:
                        if ei[group[j]] > ei[key]:
                            group[j + 1] = group[j]
                        else:
                            break
                    else:
                        break
                    j -= 1
                group[j + 1] = key
            order[lo:hi] = group
        lo = hi
    return order