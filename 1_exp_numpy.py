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
numbers_combined = np.concatenate([numbers_array, numbers_array])
print(numbers_combined)

# %%
## 20.4

# Indexing

#%%
'''
[] = is for data selection
-3, -2, -1
'''
numbers_combined[0]

# %%
numbers_combined[-2]
# %% slices -> returns an array
numbers_combined[:3]

# %%
numbers_combined[-2:]
# %%
numbers_combined[1:3]
# %%
numbers_list[4]
# %%

## Multi-dimensional arrays

#%%
numbers_array_2d = np.array([[1,2,3], [4,5,6]])

# %%
numbers_array_4d = np.array([[1,2,3], [4,5,6], [7,8,9], [10,11,12]])
numbers_array_4d_wrong = np.array([[1,2,3], [4,5,6], [7,8,9], [10,11,12, 13]])

# %% summing the right one
np.sum(numbers_array_4d)

# %% summing the wrong one
np.sum(numbers_array_4d_wrong)

# %%
4* numbers_array_4d

# %% selecting a value
numbers_array_4d[1][1]

# %% combination of selection and slicing
numbers_array_4d[0, 1:]


## Filtering arrays
# %%
numbers_array_4d <= 4
# %%
numbers_array_4d[numbers_array_4d <= 4]

# %% give odd numbers
numbers_array_4d[numbers_array_4d % 2 == 1]

# %% conditional values
np.where(numbers_array_4d < 3, "Low", "High")

# %%
ages_array = np.array([19, 21, 18, 15, 20, 30, 3, 11, 55])
age_alcohol_array = np.where(ages_array >= 21, "Yes", "No")
age_alcohol_array

# %%
np.unique(age_alcohol_array)

#%%
np.unique(age_alcohol_array, return_counts=True)

### Assignments
# %% 1.
my_array_1 = [4,7,2,9] 
my_array_2 = [10,4,4]
my_array_3 = [1,6] 

my_combined_array = np.concatenate([my_array_1, my_array_2, my_array_3])
print(my_combined_array)

#%% Assignment 2.

def greet_name(name, age):
    print(f"Hello {name}, you are {age} years old.")



#%%
greet_name("Arie", 40)

# %% define the function
def show_array_information(array):
    print(len(array))
    print(np.std(array))
    print(np.min(array))
    print(np.max(array))
    print(np.unique(array))
    print(np.unique(array, return_counts=True))

#%%
show_array_information(my_combined_array)

# %%
def return_odd_even(array):
    array_odd = array[array % 2 == 1]
    array_even = array[array % 2 != 1]
    return array_odd, array_even


#%%
return_odd_even(my_combined_array)








# %%
