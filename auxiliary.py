"""Created on Tuesday, February 21st 2023
   Author: Suraj Prakash
"""

# a function to identify missing keys and initialize them to zero 

# the way set(list) vs set(dict) behaves, there is a possible source of confusion here

def fill_params(list1: list, dict2: dict) -> dict:
    if len(list1) > len(dict2):
        for key in list(set(list1).difference(set(dict2.keys()))):
            dict2[key] = 0.0
        return dict2

    else: 
        return dict2

def reg_sq_root(x):
    pass
