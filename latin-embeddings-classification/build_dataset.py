import numpy as np
import tqdm

def build_dataset(num, converter, sents, suspect_lengths):
    size = converter.embeddings.vector_size
    x = np.zeros((num, size))
    chosen_sents = []
    for i in tqdm.tqdm(range(num)):
        done = False
        while not done:
            try:
                length = np.random.choice(suspect_lengths)
                pos = np.random.choice(len(sents) - length - 1)
                s = '. '.join(sents[pos:pos+length])
                chosen_sents.append(s)
                x[i] = converter.convert(s)
                done = True
            except IndexError:
                continue
    return x, chosen_sents
