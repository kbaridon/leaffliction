import os
import sys

import matplotlib.pyplot as plt
from PIL import Image


def main():

    try:

        if len(sys.argv) != 2:
            raise AssertionError(
                "The program should have a file path argument:"
                "./Distribution.py ./Apple"
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

        labels = list(data_dict.keys())
        values = list(data_dict.values())
        cmap = plt.get_cmap("tab20")
        bar_colors = [cmap(i % cmap.N) for i in range(len(labels))]

        plt.figure(figsize=(10, 10))

        plt.subplot(2, 1, 1)
        plt.bar(labels, values, color=bar_colors)

        plt.subplot(2, 1, 2)
        wedges, _, _ = plt.pie(values, autopct="%1.1f%%", colors=bar_colors)
        plt.legend(
            wedges,
            labels,
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            fontsize="small",
        )
        plt.title("class distribution")
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(type(e))
        print(f"error: {e}")
        return


if __name__ == "__main__":
    main()
