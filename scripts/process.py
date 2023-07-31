import argparse

from mlhousing.preprocess import init_preprocess
from mlhousing.ingest_data import fetch_housing_data
from mlhousing.train import init_training
from mlhousing.score import init_score


def parse_args():
    """Parse argument function which take command line argument for required folders and files which will be input as
    argument in different funtions

    Args:
        Command line arguments

    Return : args object

    """

    parser = argparse.ArgumentParser(
        description="Preprocessing script to preprocess raw data"
    )
    parser.add_argument(
        "-a",
        type=int,
        default=10,
    )
    parser.add_argument(
        "-b",
        type=int,
        default=10,
    )
    parser.add_argument(
        "-i",
        "--input_folder",
        type=str,
        default="data/raw",
        help="mention input_folder path to load your raw dataset",
    )
    parser.add_argument(
        "-f",
        "--input_filename",
        type=str,
        help="mention input_file name to load your raw dataset",
        default="housing.csv",
    )
    parser.add_argument(
        "-o",
        "--output_folder",
        type=str,
        help="mention output_folder path to store your preprocessed,fetaure engineered train and validation datasets",
        default="data/processed",
    )
    parser.add_argument(
        "-r",
        "--output_folder_rawdata",
        help="mention output_folder path where you want to store raw downloaded dataset",
        default="data/raw",
    )
    parser.add_argument(
        "-p",
        "--training_data_path",
        help="mention processed training and test input data path",
        default="data/processed",
    )
    parser.add_argument(
        "-m",
        "--artifacts_path",
        help="mention output path to store training artifacts",
        default="artifacts",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    """Main function to run project pipeline like data ingest, preporcess, train and score

    Args:
        None

    Return : None.

    """

    inputs = parse_args()

    fetch_housing_data(inputs.output_folder_rawdata)
    init_preprocess(inputs.input_folder, inputs.input_filename, inputs.output_folder)
    init_training(inputs.training_data_path, inputs.artifacts_path)
    init_score(inputs.training_data_path, inputs.artifacts_path)
