import os

SEED_VALUE = 42

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR,"data")

data_path = os.path.join(DATA_DIR, "exT.csv")
fpkm_data_path = os.path.join(DATA_DIR, "FPKM_gene_counts_FPKM.csv")

pickle_filename = os.path.join(DATA_DIR, "scaled_splitted_data.pickle")
fpkm_pickle_filename = os.path.join(DATA_DIR, "scaled_splitted_data_with_fpkm.pickle")

