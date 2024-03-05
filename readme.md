# BioWordVec: Improving Biomedical Word Embeddings with Subowrd Information and MeSH #
This sourcecode is a demo implementation described in the paper "BioWordVec:Improving Biomedical Word Embeddings with Subowrd Information and MeSH." This is research software, provided as is without express or implied warranties etc. see licence.txt for more details. We have tried to make it reasonably usable and provided help options, but adapting the system to new environments or transforming a corpus to the format used by the system may require significant effort. 

## Data files ##
Data: MeSH_graph.edgelist is the MeSH main-heading graph file. MeSH_dic.pkl.gz is used to align the MeSH heading ids with mention words. The PubMed corpus and MeSH RDF data can be download from NCBI. 
 
## Prerequisites ##
- python
- networkx
- gensim

## Usage ##

BioWordVec.py to automatically learn the biomedical word embedding based on PubMed text corpus and MeSH data.
