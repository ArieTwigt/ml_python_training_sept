#%% import the required modules
import pandas as pd 
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# %% import the data
df = pd.read_csv('data/german_data_clean.csv')

# %% explore the data
df.info()

# %% summarize the numeric data types
df.describe()

#%% data transformation: transform the 'response' variable
df['response'] = np.where(df['response'] == int(0), "Yes", "No")
df['response'].value_counts()

# %% explore the data with visualisations
df.plot(kind='scatter', x='credit_amount', y='age_years')

# %% elaborate data visualisation
(
    df.groupby('purpose')
    .mean()
    .reset_index()
    .plot
    .bar(x="purpose", y="credit_amount", color="red")
)


#%% specify the X and y columns
x_columns = ['age_years']
y_column = ['credit_amount']

X = df[x_columns]
y = df[y_column]

#%% split the data into a train- and test-set -> returns 4
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# %% define the algorithm
lm = LinearRegression()

# %% Train the model
lm.fit(X_train, y_train)

# %% Evaluate the model
#%% creating the table dynamically

df_preds = pd.DataFrame()

for column in x_columns:
    df_preds[column] = X_test[column]

# %% add the actual y-values from the test-set next to the predicted y-values
df_preds['actual'] = y_test
df_preds['predicted'] = lm.predict(df_preds[x_columns])
# %%
