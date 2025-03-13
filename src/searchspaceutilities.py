'''
THESE FUNCTIONS ARE TAKEN FROM
https://github.com/fei-aiart/NAS-Lung/blob/master/searchspace/search_space_utils.py

For consistency, I will use the same methods to generate the search space as was used in the NASLung project.
I have edited the removed all functions whose purpose was to prune based on latency. I will not be using latency as a constraint.
'''

import numpy as np
import copy
import time
import torch
import copy
import itertools


def get_all_search_space(min_len, max_len, channel_range):
    """
    get all configs of model

    :param min_len: min of the depth of model
    :param max_len: max of the depth of model
    :param channel_range: list, the range of channel
    :return: all search space
    """
    all_search_space = []
    # get all model config with max length
    max_array = get_search_space(max_len, channel_range)
    max_array = np.array(max_array)
    for i in range(min_len, max_len+1):
        new_array = max_array[:, :i]
        repeat_list = new_array.tolist()
        # remove repeated list from lists
        new_list = remove_repeated_element(repeat_list)
        for list in new_list:
            for first_split in range(1, i -1):
                for second_split in range(first_split + 1, i):
                    # split list
                    all_search_space.append(
                        [list[:first_split], list[first_split:second_split], list[second_split:]])
    return all_search_space


def get_search_space(max_len, channel_range, search_space=[], now=0):
    """
    Recursive.
    Get all configuration combinations

    :param max_len: max of the depth of model
    :param channel_range: list, the range of channel
    :param search_space: search space
    :param now: depth of model
    :return:
    """
    result = []
    if now == 0:
        for i in channel_range:
            result.append([i])
    else:
        for i in search_space:
            larger_channel = get_larger_channel(channel_range, i[-1])
            for m in larger_channel:
                tmp = i.copy()
                tmp.append(m)
                result.append(tmp)
    now = now + 1
    if now < max_len:
        return get_search_space(max_len, channel_range, search_space=result, now=now)
    else:
        return result


def get_larger_channel(channel_range, channel_num):
    """
    get channels which is larger than inputs

    :param channel_range: list,channel range
    :param channel_num: input channel
    :return: list,channels which is larger than inputs
    """
    result = filter(lambda x: x >= channel_num, channel_range)
    return list(result)


def remove_repeated_element(repeated_list):
    """
    Remove duplicate elements

    :param repeated_list: input list
    :return: List without duplicate elements
    """
    repeated_list.sort()
    new_list = [repeated_list[k] for k in range(len(repeated_list)) if
                k == 0 or repeated_list[k] != repeated_list[k - 1]]
    return new_list

