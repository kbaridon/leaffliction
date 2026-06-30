import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


def main():

    try:

        if len(sys.argv) != 2:
            raise AssertionError(
                "The program should have a file path argument: ./Distribution.py ./Apple"
            )

        file_path = sys.argv[1]

        if not os.path.exists(file_path):
            raise AssertionError("folder does not exists")

        data_dict = {}

        for d in os.listdir(file_path):
            if not os.path.isdir(f"{os.path.join(file_path, d)}"):
                raise AssertionError("directories should be nested")

            data_dict[d] = 0

            for f in os.listdir(f"{os.path.join(file_path, d)}"):
                with Image.open(f"{os.path.join(file_path, d, f)}") as img:
                    img.verify()
                data_dict[d] += 1

        bar_colors = ["tab:red", "tab:blue", "tab:green", "tab:orange"]

        plt.subplot(2, 1, 1)
        plt.bar(data_dict.keys(), data_dict.values(), color=bar_colors)

        data_values = list(data_dict.values())
        data_sum = 0
        for i in data_values:
            data_sum += i

        pie_labels = []
        for i in data_values:
            pie_labels.append(f"{round((i / data_sum) * 100, 1)}%")

        plt.subplot(2, 1, 2)
        plt.pie(data_dict.values(), autopct="%1.1f%%", colors=bar_colors)
        plt.title("class distribution")
        plt.show()

    except Exception as e:
        print(type(e))
        print(f"error: {e}")
        return


if __name__ == "__main__":
    main()
