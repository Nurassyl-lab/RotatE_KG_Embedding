import numpy as np

# keys names, items rows in the Fb15k entities.dict
dic_fb15k = {
    "Monaco": 10358,
    "Queen Elizabeth II": 5387,
    "Obama": 10747,
    "Michael Jordan": 9232,
    "USA": 13062,
    "Yann Tiersen (French Composer)": 5235,
    "France": 11181,
    "Bugs Bunny": 5236,
    "Gethin Creagh (New Zealand Sound engineer)": 14873,
    "Chicago Bulls": 14507,
    "Indonesia": 12506,
    "Bali": 13298,
    "Jesus Christ": 1670,
    "Nike": 4215,
    "Guitar": 13547
}

def protate_distance(emb1, emb2, C=1):
    """
    Compute the pRotatE-style distance between two entity embeddings.

    Parameters:
    - theta_e1: np.array, phase embedding of entity 1 (angles in radians)
    - theta_e2: np.array, phase embedding of entity 2 (angles in radians)
    - C: scaling factor (default=1)

    Returns:
    - Distance as a float
    """
    # C = 0.026000000536441803 
    # ! IDK what values shoud C have, I assume 1, cuz its supposed to be on the unit circle
    return 2 * C * np.linalg.norm(np.sin((emb1 - emb2) / 2), ord=2)

def cosine_similarity(vec1, vec2):
    """
    Calculates the cosine similarity between two vectors.

    Args:
        vec1 (numpy.ndarray): The first vector.
        vec2 (numpy.ndarray): The second vector.

    Returns:
        float: The cosine similarity between the two vectors.
    """
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)

    if vec1.shape != vec2.shape:
        raise ValueError("Vectors must have the same dimensions.")

    dot_product = np.dot(vec1, vec2)
    magnitude_vec1 = np.linalg.norm(vec1)
    magnitude_vec2 = np.linalg.norm(vec2)

    if magnitude_vec1 == 0 or magnitude_vec2 == 0:
        return 0.0  # Handle zero vectors

    return dot_product / (magnitude_vec1 * magnitude_vec2)

def cumm_sum(A, B, rel=None):
    if isinstance(rel, np.ndarray):
        A += rel
    A = np.rad2deg((A + np.pi) % (2 * np.pi) - np.pi)  # Keeps values in [-pi, pi]
    B = np.rad2deg((B + np.pi) % (2 * np.pi) - np.pi)  # Keeps values in [-pi, pi]
    A = A + 180
    B = B + 180


    diff = np.min((np.abs(360-(A-B)), np.abs(A-B)), axis=0)
    diff = np.cumsum(diff)[-1]
    
    return diff/360000

if __name__ == "__main__":
    # load the embeddings
    Embeddings = np.load("pRotatE_1000_Entity_Embeddings_FB15k.npy")
    Relation_DATA = np.load("pRotatE_1000_Relation_Embeddings_FB15k.npy")

    dic_distances = {}

    # metrics = "pRotatE_dist"
    # metrics = "cosine_sim"
    metrics = "cumm_sum"

    all_keys = list(dic_fb15k.keys())
    memo_pairs = []
    for key1 in all_keys[:-1]:
        for key2 in all_keys[1:]:
            index1 = dic_fb15k[key1]
            index2 = dic_fb15k[key2]

            ent1 = Embeddings[index1]
            ent2 = Embeddings[index2]

            rel = Relation_DATA[14]

            if f"{ent1}_{ent2}" not in memo_pairs and f"{ent2}_{ent1}" not in memo_pairs:
                memo_pairs.append(f"{ent1}_{ent2}")
                memo_pairs.append(f"{ent2}_{ent1}")
            else:
                continue

            # normalize entities to [-pi, pi]
            c = 0.026000000536441803 # not the same c as in the paper
            ent1 = ent1/(c/np.pi)
            rel = rel/(c/np.pi)
            ent2 = ent2/(c/np.pi)

            new_key = f"{key1} -> {key2}"
            if metrics == "pRotatE_dist":
                dist = protate_distance(ent1, ent2)
            elif metrics == "cosine_sim":
                dist = cosine_similarity(ent1, ent2)
            elif metrics == "cumm_sum":
                dist = cumm_sum(ent1, ent2, rel)
            else:
                raise ValueError("No metric!")
            dic_distances[new_key] = dist

    sorted_dict = dict(sorted(dic_distances.items(), key=lambda item: item[1], reverse=True))
    for key, value in sorted_dict.items():
        print(key, value)
