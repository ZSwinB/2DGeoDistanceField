"""Sort angular events with the historical tolerance-sensitive insertion result."""
import numpy as np
from numba import njit


@njit(cache=True)
def sort_angular_events(angles, wall_ids, count):
    order = np.argsort(angles[:count], kind='mergesort')
    left = 0
    while left < count:
        right = left + 1
        while right < count and angles[order[right]]-angles[order[right-1]] < 1e-12:
            right += 1
        repair = right-left > 1 and angles[order[right-1]] != angles[order[left]]
        if right-left > 1 and not repair:
            for q in range(left+1, right):
                if wall_ids[order[q]] < wall_ids[order[q-1]]:
                    repair = True
                    break
        if repair:
            # This comparator is nontransitive: replay the original INPUT order
            # within each connected near-angle group, not the sorted order.
            group = np.sort(order[left:right])
            for i in range(1, len(group)):
                key = group[i]
                j = i-1
                while j >= 0:
                    a1, a2 = angles[group[j]], angles[key]
                    if a1 > a2:
                        group[j+1] = group[j]
                    elif abs(a1-a2) < 1e-12:
                        if wall_ids[group[j]] > wall_ids[key]:
                            group[j+1] = group[j]
                        else:
                            break
                    else:
                        break
                    j -= 1
                group[j+1] = key
            order[left:right] = group
        left = right
    return order
