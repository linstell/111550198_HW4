from dataset import RestorationDataset


def main():
    dataset = RestorationDataset(
        degraded_dir="hw4_realse_dataset/train/degraded",
        clean_dir="hw4_realse_dataset/train/clean",
        image_size=256,
    )

    print("Number of training images:", len(dataset))

    degraded, clean = dataset[0]

    print("Degraded tensor shape:", degraded.shape)
    print("Clean tensor shape:", clean.shape)
    print("Min degraded value:", degraded.min().item())
    print("Max degraded value:", degraded.max().item())


if __name__ == "__main__":
    main()