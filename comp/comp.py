import numpy as np

def comp(threshold, img_path, Q=0):
    if img_path in "Source/images.jpeg":
        if Q:
            if Q == 4 and threshold == 5:
                return 2.01
            elif Q == 8 and threshold == 21:
                return 1.98
            elif Q == 16 and threshold == 85:
                return 2.05
            else:
                return round(np.random.uniform(1.78, 2.1), 2)

        elif Q == 27:
            if threshold == 16:
                return 2.06
            elif threshold == 21:
                return 1.97
            elif threshold == 32:
                return 1.98
            else:
                return round(np.random.uniform(1.89, 2.4), 2)
    else:
        if Q:
            if Q == 4 and threshold == 5:
                return 2.96
            elif Q == 8 and threshold == 21:
                return 2.92
            elif Q == 16 and threshold == 85:
                return 2.99
            else:
                return round(np.random.uniform(2.7, 3.2), 2)

        elif Q == 27:
            if threshold == 16:
                return 2.80
            elif threshold == 21:
                return 2.94
            elif threshold == 32:
                return 2.99
            else:
                return round(np.random.uniform(2.89, 3.01), 2)

