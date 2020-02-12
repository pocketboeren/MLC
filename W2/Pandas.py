import numpy as np
import os
import pandas as pd
import definitions

data_file = os.path.join(definitions.DATA_PATH, 'loans.csv')
pd_data = pd.read_csv(data_file, sep=';')
