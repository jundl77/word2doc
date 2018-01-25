import random
import numpy as np
from tqdm import tqdm


class DataGenerator:

    def __init__(self):
        return

    def generate_data(self, amount, num_docs):
        """Generate identical data, where the label is always the data that has the max number"""

        scores = list()
        with tqdm(total=amount) as pbar:
            for i in tqdm(range(0, amount)):
                numbers = []
                for x in range(0, 20):
                    numbers.append(random.randint(1, 1001))

                scores.append(numbers)

                pbar.update()

        return scores

    def generate_labels(self, data, num_docs):

        labels = list()
        with tqdm(total=len(data)) as pbar:
            for elem in tqdm(data):

                bins = list(map(lambda b: np.ndarray.tolist(b), np.array_split(elem, num_docs)))

                list_max = list(map(lambda l: max(l), bins))
                index = np.argmax(np.asarray(list_max))

                # 1-hot encode
                one_hot = [0, 0, 0, 0, 0]
                one_hot[index] = 1

                # Add to arrays
                labels.append(one_hot)

                pbar.update()

        return labels
