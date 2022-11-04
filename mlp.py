import numpy as np


"""
Let us note s the sigmoid function. 
For all x in R, s(x) = 1 / (1 + exp(-x))
Then, s'(x) = exp(-x) / (1 + exp(-x))^2
            = ((1 + exp(-x)) - 1) / (1 + exp(-x))^2
            = 1 / (1 + exp(-x)) - 1 / (1 + exp(-x))^2
            = (1 / (1 + exp(-x))) * (1 - 1 / (1 + exp(-x)))
            = s(x) * (1 - s(x))
Thus, s' = s * (1 - s)
"""
"""

"""
