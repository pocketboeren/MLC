import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import definitions


# Pandas has it's own data types: DataFrame and Series

data_file = os.path.join(definitions.DATA_PATH, 'loans.csv')
pd_data = pd.read_csv(data_file, sep=';')
# other options:
#   usecols = cols
#   na_values = ['..']
#   header = 3
#   comment = '#'

pd_data.head()
pd_data.tail()

labels_example = ['customer_id2', 'Limit2', 'Outstanding2', 'Arrears2', 'PD', 'EAD', 'LGD']
pd_data.columns = labels_example

# Series
pd_data['customer_id2'].head()
# DataFrame
pd_data[['customer_id2']].head()
print('Shape is: ', pd_data.shape)

type(pd_data['customer_id2'][0:3])
pd_data_rows = pd_data[0:10]

# loc & iloc
rows = [1, 3]
columns = ['customer_id2', 'Limit2']
selection = pd_data.loc[rows, columns]

row0 = pd_data.iloc[0]
row3 = pd_data.iloc[3]
row3_and_7 = pd_data.iloc[[3, 7]]
column_cus_2 = pd_data.loc[:, ['customer_id2']]

# Save the cleaned up DataFrame to a CSV file without the index
save_data_file = os.path.join(definitions.DATA_PATH, 'loans2.csv')
pd_data.to_csv(save_data_file, index=False)

list_keys = ['Country', 'Total']
list_values = [['United States', 'Soviet Union', 'United Kingdom'], [1118, 473, 273]]
zipped = list(zip(list_keys, list_values))
print(zipped)
data = dict(zipped)
print(data)
df = pd.DataFrame(data)
print(df)


# plotting with pandas
pd_data[['customer_id2', 'Limit2']].plot()
plt.title('Temperature in Austin')
plt.xlabel('Hours since midnight August 1, 2010')
plt.ylabel('Temperature (degrees F)')
pd_data[['customer_id2', 'Limit2']].plot(subplots=True)

# selection
sel = pd_data['Limit2'] > 4
pd_data.loc[sel, 'customer_id2'] = np.nan


# apply and map
def example_function(x):
    return 5 * x


pd_data_transform = pd_data[['Limit2']].apply(example_function)


# Create the dictionary: red_vs_blue
class SmartDict(dict):
    def __missing__(self, key):
        return 'Def'


map_stage = {5: 'A', 6: 'B'}
map_stage = SmartDict(map_stage)

# Use the dictionary to map the 'winner' column to the new column: election['color']
pd_data['stage'] = pd_data['Limit2'].map(map_stage)
pd_data['stage'] = pd_data['Limit2'].map(map_stage).fillna('C')

pd_data = pd_data.drop(columns=['stage'])
pd_data['stage'] = pd_data['Limit2'].map(map_stage)

pd_data['Limit2'].describe()
pd_data['Limit2'].plot(kind='box')
pd_data['Limit2'].quantile([0.1])
pd_data.plot(kind='scatter', x='Outstanding2', y='Limit2')
pd_data['Outstanding2'].plot(kind='hist', density=1, range=(0,30))

