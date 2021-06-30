"""Locate most called functions"""
import numpy as np


def _load_code(filename):
    with open(filename, "r") as fin:
        lines = fin.readlines()
        lines = [line.strip() for line in lines]
    # get functions
    funcs = dict()
    for i, line in enumerate(lines):
        if line.strip().startswith("def"):
            funcs[line.split("def")[1].split("(")[0].strip()] = {"location":i}

    func_locations = np.array([funcs[func]["location"] for func in funcs])
    funcs_lenths = func_locations[1:] - func_locations[:-1]
    for i, line in enumerate(lines[func_locations[-1]:]):
        if "return" in line:
            break
    funcs_lenths = np.append(funcs_lenths, i+1)


    for j, func in enumerate(funcs):
        dict_func = {"location":funcs[func]["location"],
                     "called":0, "callers":{}, "length":funcs_lenths[j]}
        for i, line in enumerate(lines):
            if func in line and "def" not in line:
                caller = _get_surounding_idx(func_locations, i)
                caller = list(funcs.keys())[caller]
                dict_func["called"] += 1
                if caller not in dict_func["callers"]:
                    dict_func["callers"][caller] = 0
                dict_func["callers"][caller] += 1
        funcs[func] = dict_func.copy()
    return funcs


def _get_surounding_idx(indexes, ind):
    signs = (np.sign(indexes-ind)+1)//2
    if 1 in signs:
        index_ = list(signs).index(1)
        if index_ > 0:
            index_ -= 1
    else:
        index_ = -1

    # index_ = indexes[index_]
    return index_

def main():
    """main"""
    funcs = _load_code("SWE_laggedPF.py")

    func_names = [func for func in funcs]
    func_call_num = [funcs[func]["called"] for func in funcs]
    order = np.flipud(np.argsort(func_call_num))
    func_call_num = np.array(func_call_num)[order]
    func_names = np.array(func_names)[order]
    print(func_names)
    print(func_call_num)

    # for func in funcs:
    #     print(func)
    #     for key in funcs[func]:
    #         print("\t ", key, funcs[func][key])
if __name__ == '__main__':
    main()
