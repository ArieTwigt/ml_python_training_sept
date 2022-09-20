#%% import the required modules
import pandas as pd

#%% import the data set
data = pd.read_csv("data/german_data_clean.csv")

## Exploring the data set

# %% dimensions of the data frame
data.shape
# %% column names of the data frame
data.columns

# %% technical information of the data frame
data.info()

#%% statistical information of the data frame
data.describe()

# %%
'''
Data types

* str: "Arie"
* int: 45
* float: 1.79
* bool: True

'''

#%% show the first rows of the data set
data.head()
# %% show the latest rows of the data set
data.tail()

## Selections in a DataFrame

# %%
data.head()
# %% show the first row
data.iloc[0]
# %% sub selections from the data frame
data.iloc[[3, 7, 198, 500], [3,-1]]

## Selecting columns
# %%
data.columns
# %%
data['savings']
# %%
data['credit_amount'] * 10

# %%
selected_colums = ['purpose', 'credit_amount', 'age_years', 'response']

data_selected_columns = data[selected_colums]
# %%
data_selected_columns


## Applying calculations in Pandas

# %%
data['credit_amount'].sum()
# %%
data['duration_months'].mean()

# %%
data['monthly_payment'] =  data['credit_amount'] / data['duration_months']