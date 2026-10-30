"""Screen by dHash, then group only candidates with low RGB thumbnail error."""
import numpy as np


def group_similar(rows, thumbnails, max_distance=4, max_rmse=0.025):
    """Coarse fundus shape alone is not a duplicate; record all candidates for review."""
    parent = list(range(len(rows)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    candidates = []
    hashes = [int(row['dhash'], 16) for row in rows]
    for i in range(len(rows)):
        for j in range(i):
            distance = (hashes[i] ^ hashes[j]).bit_count()
            if distance > max_distance:
                continue
            a = thumbnails[rows[i]['sample_id']]
            b = thumbnails[rows[j]['sample_id']]
            rmse = float(np.sqrt(np.mean((a - b) ** 2)))
            grouped = rmse <= max_rmse
            if grouped:
                parent[find(i)] = find(j)
            candidates.append({
                'a': rows[j]['sample_id'], 'b': rows[i]['sample_id'],
                'hamming': distance, 'rgb_rmse': rmse, 'grouped': grouped,
            })
    for i, row in enumerate(rows):
        row['similarity_group'] = rows[find(i)]['sample_id']
    return candidates
