# test_train.py
from gan.train_gan import train_gan



if __name__ == "__main__":
    train_gan(
        data_path="data/crops",
        epochs=60
    )