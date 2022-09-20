#%% 
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, roc_curve, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt

#%% import the dataset
df = pd.read_csv("data/german_data_clean.csv")

# %%
df.head()

# %%
df['response'].value_counts()

# %%
x_columns = ['age_years', 'credit_amount', 'duration_months', 'purpose', 'property', 'credit_history', 'housing', 'job']
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

# %% visualize the confustion matrix
ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues', fmt='g')
ax.set_title(f"Confusion Matrix total: ({len(y_test)})")
ax.set_xlabel("Predicted values")
ax.set_ylabel("Actual values")

# %% calculate the metrics
model_accuracy = accuracy_score(y_test, y_pred) # (199 + 20) / (300)
model_precision = precision_score(y_test, y_pred) # (20 / (20 + 15)) how serious can we take a 'positive' prediction from the model
model_recall = recall_score(y_test, y_pred, pos_label=1) # (20 / 86) how good is our model in identifying the positive class
model_specificity = recall_score(y_test, y_pred, pos_label=0) # (199 / (199 + 15)) how good is our model in identifiying the negative class


# %%  create the ROC-curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1]) # function returns 3 values/objects, the '_' is a placeholder

#%%
current_auc_score = roc_auc_score(y_test, y_pred_proba[:, 1])

# %%
plt.plot(fpr, tpr, label=f"AUC {str(current_auc_score)}")
plt.plot([0,1], [0,1], linestyle="--", lw=2, color='r', label='Random guess')
plt.legend(loc='lower right')

# %%
