# Determination of Classification on Cancer Genomes
Exploring the explainability of neurons indentifying the cancerous tissues. 

Dataset: [The Cancer Genome Atlas Program and TCGA dataset](https://www.cancer.gov/about-nci/organization/ccg/research/structural-genomics/tcga)

# Dependencies
```
Python = 3.6+
Tensorflow = 2.1
Scikit-Learn
Numpy 
Pandas
Scipy
Matplotlib
Lime
Tensorboard
Seaborn
Keras==2.3.1
Jupyter
h5py
```
and more...

# Running the project
- Add data to train the model to `data/` directory. 
- Command to train the model and extract explainations.
```python
python -m src.classification
```
All the configs and options are available in the file [src/classification.py](src/classification.py)
