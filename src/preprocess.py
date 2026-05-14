import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


def load_data(file_path):

    df = pd.read_csv(file_path)

    return df


def preprocess_data(df):

    # Drop unnecessary column
    df.drop("StudentID", axis=1, inplace=True)

    # Label Encoding
    le_extra = LabelEncoder()

    le_training = LabelEncoder()

    le_target = LabelEncoder()

    df['ExtracurricularActivities'] = (
        le_extra.fit_transform(
            df['ExtracurricularActivities']
        )
    )

    df['PlacementTraining'] = (
        le_training.fit_transform(
            df['PlacementTraining']
        )
    )

    df['PlacementStatus'] = (
        le_target.fit_transform(
            df['PlacementStatus']
        )
    )

    # Save cleaned dataset
    df.to_csv(
        "data/processed/cleaned_placement_data.csv",
        index=False
    )

    # Feature Target Split
    X = df.drop('PlacementStatus', axis=1)

    y = df['PlacementStatus']

    # Save feature column order
    feature_columns = X.columns.tolist()

    # Scaling
    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.2,
        random_state=42
    )

    # Save preprocessing artifacts
    joblib.dump(
        scaler,
        "model/scaler.pkl"
    )

    joblib.dump(
        le_extra,
        "model/le_extra.pkl"
    )

    joblib.dump(
        le_training,
        "model/le_training.pkl"
    )

    joblib.dump(
        le_target,
        "model/le_target.pkl"
    )

    joblib.dump(
        feature_columns,
        "model/feature_columns.pkl"
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test
    )