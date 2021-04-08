import argparse

from utils import validate_path
from data import DataSet
from train import Trainer


def main():
    parser = argparse.ArgumentParser(description="Training DCGAN on CelebA dataset")
    parser.add_argument("--checkpoint_dir", type=str, default="./model/checkpoint", help="Path to write checkpoint")
    parser.add_argument("--progress_dir", type=str, default="./data/face_gan", help="Path to write training progress image")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to dataset")
    parser.add_argument("--latent_dim", type=int, default=100, help="Latent space dimension")
    parser.add_argument("--test_size", type=int, default=4, help="Square root number of test images to control training progress")
    parser.add_argument("--batch_size", type=int, default=100, help="Number of training steps per epoch")
    parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs for training")
    
    args = vars(parser.parse_args())
    
    validate_path(args["checkpoint_dir"])
    validate_path(args["progress_dir"])
    
    datagen = DataSet(args["dataset_dir"])
    dataset, total_steps = datagen.build(batch_size=args["batch_size"])
    
    DCGAN = Trainer(progress_dir=args["progress_dir"],
                    checkpoint_dir=args["checkpoint_dir"],
                    z_dim=args["latent_dim"],
                    test_size=args["test_size"],
                    batch_size=args["batch_size"],
                    learning_rate=args["lr"])
    
    DCGAN.train_loop(dataset=dataset,
                     epochs=args["epochs"],
                     total_steps=total_steps)
    
if __name__ == "__main__":
    main()