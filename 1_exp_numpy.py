#%% import the required modules
import numpy as np


#%%
numbers_list = [1, 2, 3, 4, 5]

#%%
numbers_list + numbers_list

# %%
numbers_array = np.array([6,7,8,9,0])

# %%
result_array = numbers_array + numbers_array

# %%
numbers_array * numbers_array

# %%
4 * numbers_array

# statistical functions
# %%
np.sum(numbers_array)

# %%
np.mean(numbers_array)


# %% concatenate arrays
combined_array = np.concatenate([numbers_array, numbers_array])
print(combined_array)

# %%
