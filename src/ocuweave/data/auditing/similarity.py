"""Conservative perceptual grouping; similarity is not evidence of patient identity."""
def group_similar(rows, max_distance=4):
    parent=list(range(len(rows)))
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]];i=parent[i]
        return i
    candidates=[]
    hashes=[int(r['dhash'],16) for r in rows]
    for i in range(len(rows)):
        for j in range(i):
            distance=(hashes[i]^hashes[j]).bit_count()
            if distance<=max_distance:
                parent[find(i)]=find(j)
                candidates.append({'a':rows[j]['sample_id'],'b':rows[i]['sample_id'],'hamming':distance})
    for i,row in enumerate(rows):row['similarity_group']=rows[find(i)]['sample_id']
    return candidates
