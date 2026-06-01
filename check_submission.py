import numpy as np


def main():
    data = np.load("outputs/pred.npz")

    print("Number of images:", len(data.files))
    print("First 10 keys:", data.files[:10])

    first_key = data.files[0]
    first_image = data[first_key]

    print("First key:", first_key)
    print("First image shape:", first_image.shape)
    print("First image dtype:", first_image.dtype)
    print("Min value:", first_image.min())
    print("Max value:", first_image.max())


if __name__ == "__main__":
    main()