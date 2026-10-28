FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir mlflow scikit-learn pandas

COPY . /app

ENV MLFLOW_TRACKING_URI=sqlite:////app/mlflow.db

EXPOSE 5001

CMD ["mlflow", "models", "serve", "-m", "models:/Diabetes-Prediction-Model/1", "-h", "0.0.0.0", "-p", "5001", "--env-manager", "local"]