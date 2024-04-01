Alexander Glynn's Senior Thesis Project at Harvard University, presenting and analyzing a method (Conceptually Diverse Image) of producing fairer results in vector database image retrieval.

Structure:

Main Folder:

helpers.py: useful utility functions
image_database: implementation of the vector database class and retrieval algorithms

learn_MI_order.ipynb: learning mutual information between CLIP embeddings and sensitive attributes, as presented in Wang et al. 2021

experiments_gender_in_image_search.ipynb: experiments on the Gender in Image Search Dataset (https://github.com/mjskay/gender-in-image-search)
experiments_occupations_2.ipynb: experiments on the Occupations Dataset collected in (https://arxiv.org/abs/1901.10265)
experiments_celebA.ipynb: experiments on the CelebA dataset (https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

plots.ipynb: turning experimental results into visualizations

results/: holds raw .pkl data files with results of experiments, as well as graphics used in the thesis.