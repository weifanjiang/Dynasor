import math
from collections import Counter, defaultdict
from typing import List
import numpy as np
from evaluator import extract_answer, strip_string, math_equal, extract_first_boxed_answer


def entropy(Plist):
    if len(Plist):
        result = 0
        for x in Plist:
            result += (-x) * math.log(x, 2)
        return result
    else:
        return 0


def norm(Olist):
    s = sum(Olist)
    return [o / s for o in Olist]


def count(Olist):
    x_dict = defaultdict(lambda: 0.0)
    for x in Olist:
        x_dict[x] += 1
    cc = [c for _, c in x_dict.items()]
    # print(cc)
    return cc


def item_entropy(answers: List) -> float:
    return entropy(norm(count(answers)))


def count_not_empty(answers):
    return sum(1 for answer in answers if answer != '')


def eqaul_group(answers):
    equiv_classes = []

    for answer in answers:
        weight = 1
        flag = 0
        for i, rep in enumerate(equiv_classes):
            if math_equal(answer, rep):
                flag = 1
                break
        if flag:
            continue
        equiv_classes.append(answer)

    return len(equiv_classes) == 1


def majority_voting(answers):
    equiv_classes = []
    equiv_weights = []
    max_vote = 0
    for answer in answers:
        weight = 1
        flag = 0
        for i, rep in enumerate(equiv_classes):
            if math_equal(answer, rep):
                flag = 1
                equiv_weights[i] = equiv_weights[i] + weight
                if equiv_weights[i] > max_vote:
                    max_vote = equiv_weights[i]
                    max_rep = answer
                break
        if flag:
            continue
        equiv_classes.append(answer)
        equiv_weights.append(weight)
        if max_vote == 0:
            max_vote = weight
            max_rep = answer
    return max_rep


def obtaint_answer(s):
    # Find first unpaired } by counting { and }
    stack = []
    for i, c in enumerate(s):
        if c == '{':
            stack.append(c)
        elif c == '}':
            if not stack:  # No matching { found
                return s[:i]
            stack.pop()
    return ""