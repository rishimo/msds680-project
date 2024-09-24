import pandas as pd
import spacy
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import mlflow
from metaflow import conda_base, kubernetes, FlowSpec, step, Parameter
from utils import *


# @conda_base(
#     packages={
#         "scikit-learn": "1.5.1",
#         "pandas": "2.1.4",
#         "spacy": "3.4.0",
#         "mlflow": "2.15.1",
#     },
#     python="3.10",
# )
class TrainModelFlow(FlowSpec):
    test_size = Parameter(
        "test_size", default=0.2, help="Proportion of data to reserve for testing"
    )
    random_state = Parameter(
        "random_state", default=42, help="Random state for reproducibility"
    )
    model_name = Parameter(
        "model_name",
        default="best_model",
        help="Name to register the model under in MLFlow",
    )
    data_file = Parameter(
        "data_file",
        default="../data/winemag-data-130k-v2.csv",
        type=str,
        required=True,
    )

    @step
    def start(self):
        """
        Setup step to download the spaCy model if necessary.
        """
        import subprocess

        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
        self.next(self.setup)

    @step
    def setup(self):
        """
        Load the dataset and split into train/test sets.
        """
        # Load spaCy language model
        self.nlp = spacy.load("en_core_web_sm")

        # Load dataset and exclude the first column ('Unnamed: 0')
        self.data = pd.read_csv(self.data_file).drop(columns=["Unnamed: 0"])

        # Separate features and target
        self.features = self.data.drop(columns=["points"])
        self.target = self.data["points"]

        # Split the data into training and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.features,
            self.target,
            test_size=self.test_size,
            random_state=self.random_state,
        )
        self.next(self.feature_transformation)

    @kubernetes()
    @step
    def feature_transformation(self):
        """
        Apply spaCy preprocessing to the description and encode other categorical features.
        """
        # Apply preprocessing to 'description' in training and test sets
        self.X_train["description"] = self.X_train["description"].apply(
            lambda x: spacy_preprocess(x, self.nlp)
        )
        self.X_test["description"] = self.X_test["description"].apply(
            lambda x: spacy_preprocess(x, self.nlp)
        )
        print("here")
        # Define the column transformer
        self.preprocessor = ColumnTransformer(
            transformers=[
                (
                    "tfidf",
                    TfidfVectorizer(max_features=10),
                    "description",
                ),  # TF-IDF on 'description'
                (
                    "onehot",
                    OneHotEncoder(handle_unknown="ignore"),
                    ["country", "province", "winery", "variety"],
                ),  # OneHotEncode other categorical features
                (
                    "scaler",
                    StandardScaler(),
                    ["price"],
                ),  # Standardize numerical features like 'price'
            ],
            remainder="drop",  # Drop remaining columns that are not processed
        )

        # Fit the preprocessor and transform the training and test sets
        self.X_train_transformed = self.preprocessor.fit_transform(self.X_train)
        self.X_test_transformed = self.preprocessor.transform(self.X_test)

        self.next(self.train_model)

    @kubernetes()
    @step
    def train_model(self):
        """
        Train a Linear Regression model and calculate the MSE on the test set.
        """
        mlflow.set_tracking_uri("https://mlflow-serv-617145385837.us-west2.run.app")
        mlflow.set_experiment("Wine_Reviews_Experiment")
        with mlflow.start_run():
            # Train the Linear Regression model using transformed features
            self.model = LinearRegression()
            self.model.fit(self.X_train_transformed, self.y_train)

            # Predict on the test set
            self.y_pred = self.model.predict(self.X_test_transformed)
            self.mse = mean_squared_error(self.y_test, self.y_pred)

            print(f"Model Mean Squared Error (MSE): {self.mse}")
            mlflow.log_param("random_state", self.random_state)
            mlflow.log_param("test_size", self.test_size)
            mlflow.log_metric("mse", self.mse)

            # Log the model and the preprocessor
            mlflow.sklearn.log_model(self.model, self.model_name)
            mlflow.sklearn.log_model(
                self.preprocessor, self.model_name + "_preprocessor"
            )

            # register the model
            mlflow.register_model(
                f"runs:/{mlflow.active_run().info.run_id}/{self.model_name}",
                self.model_name,
            )
            mlflow.register_model(
                f"runs:/{mlflow.active_run().info.run_id}/{self.model_name}_preprocessor",
                self.model_name + "_preprocessor",
            )

            mlflow.end_run()

            print(f"Model registered in MLFlow as {self.model_name}")

        self.next(self.end)

    @step
    def end(self):
        """
        End of the training flow.
        """
        print("Training flow complete.")


if __name__ == "__main__":
    TrainModelFlow()
