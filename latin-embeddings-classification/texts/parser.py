import os
import re


def lines_from_file(fname):
    """The following code is modified from the 'parse' function contained at [1].

    [1] https://github.com/QuantitativeCriticismLab/PNAS_2017_QuantitativeCriticism/blob/master/Code/usefulfunctions.py
    """
    with open(fname, 'r') as f:
        text = f.read()
    lines = text.split("<milestone")
    parsed = []
    for ll in lines[1:]:
        line = ll[ll.index(">")+1:]
        if not line:
	        continue
        if "<" in line :
            line = re.sub('<[^>]*>', '', line).strip()
        if len(line) != 0:
            line = re.sub('([A-Z][A-Za-z]?[.])', '', line).replace('  ', ' ')
        parsed.append(line)
    return parsed

for fname in os.listdir('livy'):
  lines = lines_from_file(os.path.join('livy', fname))
  with open(os.path.join('parsed-livy', fname), 'w') as f:
      text = " ".join(" ".join([l for l in lines if 'periocha libri' not in l]).split())
      f.write(text)

