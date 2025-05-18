import os
import pickle
import click
import pandas as pd
import logging

from sklearn.feature_extraction import DictVectorizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def dump_pickle(obj, filename: str):
    with open(filename, "wb") as f_out:
        return pickle.dump(obj, f_out)


def read_dataframe(filename: str):
    df = pd.read_parquet(filename)

    df['duration'] = df['lpep_dropoff_datetime'] - df['lpep_pickup_datetime']
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)
    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    return df


def preprocess(df: pd.DataFrame, dv: DictVectorizer, fit_dv: bool = False):
    logger.info("First 5 rows of input DataFrame to preprocess:\n%s", df.head())
    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']
    logger.info(f"Number of unique PU_DO pairs: {df['PU_DO'].nunique()}")
    logger.info(f"First 10 unique PU_DO pairs: {df['PU_DO'].unique()[:10]}")
    categorical = ['PU_DO']
    numerical = ['trip_distance']
    # Ensure target is not in features
    features = categorical + numerical
    if 'tip_amount' in features:
        features.remove('tip_amount')
    dicts = df[features].to_dict(orient='records')
    if fit_dv:
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)
    logger.info("First 5 rows of vectorized X:\n%s", X[:5])
    logger.info(f"Vectorized X shape: {X.shape}")
    return X, dv


@click.command()
@click.option(
    "--raw_data_path",
    help="Location where the raw NYC taxi trip data was saved"
)
@click.option(
    "--dest_path",
    help="Location where the resulting files will be saved"
)
def run_data_prep(raw_data_path: str, dest_path: str, dataset: str = "green"):
    # Load parquet files
    df_train = read_dataframe(
        os.path.join(raw_data_path, f"{dataset}_tripdata_2022-01.parquet")
    )
    df_val = read_dataframe(
        os.path.join(raw_data_path, f"{dataset}_tripdata_2022-02.parquet")
    )
    df_test = read_dataframe(
        os.path.join(raw_data_path, f"{dataset}_tripdata_2022-03.parquet")
    )

    # Extract the target
    target = 'tip_amount'
    y_train = df_train[target].values
    y_val = df_val[target].values
    y_test = df_test[target].values

    # Fit the DictVectorizer and preprocess data
    dv = DictVectorizer()
    X_train, dv = preprocess(df_train, dv, fit_dv=True)
    X_val, _ = preprocess(df_val, dv, fit_dv=False)
    X_test, _ = preprocess(df_test, dv, fit_dv=False)

    # Create dest_path folder unless it already exists
    os.makedirs(dest_path, exist_ok=True)

    # Save DictVectorizer and datasets
    dump_pickle(dv, os.path.join(dest_path, "dv.pkl"))
    dump_pickle((X_train, y_train), os.path.join(dest_path, "train.pkl"))
    dump_pickle((X_val, y_val), os.path.join(dest_path, "val.pkl"))
    dump_pickle((X_test, y_test), os.path.join(dest_path, "test.pkl"))


if __name__ == '__main__':
    run_data_prep()
