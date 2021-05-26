# Profiling of Intertextuality in Latin Literature Using Word Embeddings

This repository includes the data and code relevant to the paper Burns, Brofos, Li, Chaudhuri, and Dexter 2021, ["Profiling of Intertextuality in Latin Literature Using Word Embeddings"](https://www.aclweb.org/anthology/2021.naacl-main.389/) in *Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies*.

There are three subsections to this repository:

- [latin-embeddings-comparison](https://github.com/QuantitativeCriticismLab/NAACL-HLT-2021-Latin-Intertextuality/tree/main/latin-embeddings-classification) (cf. Sect. 2 "Evaluation and optimization of word embedding models for Latin")
- [latin-embeddings-search](https://github.com/QuantitativeCriticismLab/NAACL-HLT-2021-Latin-Intertextuality/tree/main/latin-embeddings-search) (cf. Sect. 4.1 "Enhanced intertextual search")
- [latin-embeddings-classification](https://github.com/QuantitativeCriticismLab/NAACL-HLT-2021-Latin-Intertextuality/tree/main/latin-embeddings-comparison) (cf. Sect. 4.2 "Anomaly detection" )

Tested on Python 3.7.10. Dependencies necessary to run the experiments are included in ```requirements.txt```. Specific instructions to replicate the experiments is given in the ```README.md``` (and corresponding code notebooks) in each subfolder. 

## Abstract

Intertextual relationships between authors is of central importance to the study of literature. We report an empirical analysis of intertextuality in classical Latin literature using word embedding models. To enable quantitative evaluation of intertextual search methods, we curate a new dataset of 945 known parallels drawn from traditional scholarship on Latin epic poetry. We train an optimized word2vec model on a large corpus of lemmatized Latin, which achieves state-of-the-art performance for synonym detection and outperforms a widely used lexical method for intertextual search. We then demonstrate that training embeddings on very small corpora can capture salient aspects of literary style and apply this approach to replicate a previous intertextual study of the Roman historian Livy, which relied on hand-crafted stylometric features. Our results advance the development of core computational resources for a major pre-modern language and highlight a productive avenue for cross-disciplinary collaboration between the study of literature and NLP.

## Acknowledgments

This work was conducted under the auspices of the Quantitative Criticism Lab (www.qcrit.org), an interdisciplinary group co-directed by P.C. and J.P.D. and supported by an American Council of Learned Societies Digital Extension Grant and a National Endowment for the Humanities Digital Humanities Advancement Grant (Grant No. HAA-271822-20). P.C. was supported by a Mellon New Directions Fellowship, and J.P.D. by a Neukom Fellowship and a Harvard Data Science Fellowship. The material contributed by J.A.B. is based upon work supported by the National Science Foundation Graduate Research Fellowship under Grant No.1752134. Any opinion, findings, and conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect the views of the National Science Foundation. We thank Adriana Cásarez, James Patterson, and Ariane Schwartz for assistance with compiling the Valerius Flaccus intertextuality dataset, and Tommaso Spinelli for sharing the dictionary of Latin near-synonyms.

## Citation
```
@inproceedings{burns-etal-2021-profiling,
  title = "Profiling of Intertextuality in {L}atin Literature Using Word Embeddings",
  author = "Burns, Patrick J. and
            Brofos, James A. and
            Li, Kyle and
            Chaudhuri, Pramit and
            Dexter, Joseph P.",
  booktitle = "Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
  month = jun,
  year = "2021",
  address = "Online",
  publisher = "Association for Computational Linguistics",
  url = "https://www.aclweb.org/anthology/2021.naacl-main.389",
  pages = "4900--4907",
  abstract = "Identifying intertextual relationships between authors is of central importance to the study of literature. We report an empirical analysis of intertextuality in classical Latin literature using word embedding models. To enable quantitative evaluation of intertextual search methods, we curate a new dataset of 945 known parallels drawn from traditional scholarship on Latin epic poetry. We train an optimized word2vec model on a large corpus of lemmatized Latin, which achieves state-of-the-art performance for synonym detection and outperforms a widely used lexical method for intertextual search. We then demonstrate that training embeddings on very small corpora can capture salient aspects of literary style and apply this approach to replicate a previous intertextual study of the Roman historian Livy, which relied on hand-crafted stylometric features. Our results advance the development of core computational resources for a major premodern language and highlight a productive avenue for cross-disciplinary collaboration between the study of literature and NLP.",
}
```