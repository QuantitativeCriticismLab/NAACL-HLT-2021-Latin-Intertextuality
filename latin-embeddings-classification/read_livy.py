import os

def read_livy():
    parseddir = os.path.join('texts', 'parsed-livy')
    docs = []
    for fname in os.listdir(parseddir):
        with open(os.path.join(parseddir, fname), 'r') as f:
            docs.append(f.read().lower())
    docs = ' '.join(docs)
    sents = docs.split('.')
    return [s.strip() for s in sents if len(s) > 10]

