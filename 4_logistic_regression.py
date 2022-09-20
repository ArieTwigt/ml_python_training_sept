#%% 
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

#%% import the dataset
df = pd.read_csv("data/german_data_clean.csv")

# %%
df.head()

# %%
df['response'].value_counts()

# %%
x_columns = ['age_years', 'credit_amount', 'duration_months', 'purpose', 'property']
y_column = ['response']

#%% create a data frame that only contains the x-variables
df_x = pd.get_dummies(df[x_columns],)

# %%
x_columns_dummies = list(df_x.columns)
X = df_x[x_columns_dummies]

y = df[y_column]

# %% split the data set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)


# %% training the model
model = LogisticRegression()
model.fit(X_train, y_train)

# %%
model.score(X_test, y_test)

# %%
y_pred = model.predict(X_test)

# %%
y_pred_proba = model.predict_proba(X_test)

# %%
df_preds = pd.DataFrame()

for column in x_columns_dummies:
    df_preds[column] = X_test[column]

# %%
df_preds['actual'] = y_test
df_preds['predicted'] = model.predict(df_preds[x_columns_dummies])

# %%
y_pred = df_preds['predicted']

#%% create confusion matrix
cf_matrix = confusion_matrix(y_test, y_pred)

# %%
