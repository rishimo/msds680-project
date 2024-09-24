from metaflow import FlowSpec, step, Parameter, conda_base
import mlflow
import pandas as pd
import spacy
from utils import *


# Conda environment with necessary packages
# @conda_base(
#     packages={
#         "scikit-learn": "1.5.1",
#         "pandas": "2.1.4",
#         "spacy": "3.7.6",
#         "mlflow": "2.15.1",
#     }
# )
class ScoringFlow(FlowSpec):
    model_name = Parameter("model_name", default="best_model")
    new_data_file = Parameter("new_data_file", default="../data/winemag-subset.csv")

    @step
    def start(self):
        """
        Setup step to download the spaCy model if necessary.
        """
        import subprocess

        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
        self.nlp = spacy.load("en_core_web_sm")
        self.next(self.setup)

    @step
    def setup(self):
        """
        Load new data and preprocess it.
        """

        # Load new data
        self.new_data = pd.read_csv(self.new_data_file)

        self.new_data["description"] = self.new_data["description"].apply(
            lambda x: spacy_preprocess(x, self.nlp)
        )

        self.next(self.load_model)

    @step
    def load_model(self):
        """
        Load the preprocessor and model from MLFlow.
        """
        mlflow.set_tracking_uri("https://mlflow-serv-617145385837.us-west2.run.app")

        # Load the preprocessor and the model
        preprocessor_uri = f"models:/{self.model_name}_preprocessor/Production"
        model_uri = f"models:/{self.model_name}/Production"

        self.preprocessor = mlflow.sklearn.load_model(preprocessor_uri)
        self.model = mlflow.sklearn.load_model(model_uri)

        self.next(self.predict)

    @step
    def predict(self):
        """
        Apply the preprocessor to the new data and make predictions.
        """
        # Apply the preprocessor (this includes TF-IDF, OneHotEncoder, and StandardScaler)
        self.new_data_transformed = self.preprocessor.transform(self.new_data)

        # Make predictions using the loaded model
        self.predictions = self.model.predict(self.new_data_transformed)

        self.next(self.end)

    @step
    def end(self):
        """
        Output predictions.
        """
        print("Predictions on new data:")
        print(self.predictions)


if __name__ == "__main__":
    ScoringFlow()
