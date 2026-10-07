# End-to-End MLOps Heart Disease Prediction

## 1. Project Overview

This project implements an end-to-end Machine Learning Operations (MLOps) workflow for predicting the presence or absence of heart disease using the UCI Heart Disease dataset.

The project demonstrates the complete machine learning lifecycle, including:

- Data acquisition
- Exploratory Data Analysis (EDA)
- Data preprocessing
- Feature engineering
- Model development
- Cross-validation
- Hyperparameter tuning
- MLflow experiment tracking
- Model packaging
- Automated testing
- CI/CD using GitHub Actions
- FastAPI model serving
- Docker containerization
- Kubernetes deployment using Minikube
- Application monitoring using Prometheus
- Application logging

---

## 2. Dataset

The project uses the **Heart Disease UCI Dataset**.

The dataset is downloaded programmatically using the `ucimlrepo` Python package.

The data acquisition script is:

```text
src/download_data.py
```

### Dataset Details

- Number of records: 303
- Original number of columns: 14
- Original target column: `num`

The original target contains values from 0 to 4.

For binary classification, the target was transformed as follows:

```text
0       -> No Heart Disease
1,2,3,4 -> Heart Disease Present
```

The new binary target column is named:

```text
target
```

### Class Distribution

| Target | Meaning | Records |
|---|---|---:|
| 0 | No Heart Disease | 164 |
| 1 | Heart Disease | 139 |

The corresponding percentages are approximately:

- No Heart Disease: 54.13%
- Heart Disease: 45.87%

The target classes are therefore reasonably balanced.

---

## 3. Data Quality Analysis

Missing-value analysis identified:

| Feature | Missing Values |
|---|---:|
| `ca` | 4 |
| `thal` | 2 |

All other columns contained no missing values.

The dataset contained:

```text
Duplicate rows: 0
```

Missing values are handled inside the machine-learning preprocessing pipeline to avoid data leakage.

---

## 4. Exploratory Data Analysis

Exploratory Data Analysis was performed using Pandas, Matplotlib, and Seaborn.

The EDA includes:

- Dataset structure inspection
- Summary statistics
- Missing-value analysis
- Duplicate analysis
- Target class distribution
- Numerical feature histograms
- Correlation analysis
- Correlation heatmap
- Feature-target correlation analysis

The EDA notebook is located at:

```text
notebooks/01_eda.ipynb
```

Generated visualizations are stored in:

```text
screenshots/
```

Important EDA visualizations include:

```text
class_balance.png
numerical_histograms.png
correlation_heatmap.png
```

### Strongest Absolute Target Correlations

The strongest observed absolute correlations with the binary target were:

| Feature | Absolute Correlation |
|---|---:|
| thal | 0.5257 |
| ca | 0.4604 |
| exang | 0.4319 |
| oldpeak | 0.4245 |
| thalach | 0.4172 |
| cp | 0.4144 |

Correlation analysis was used for exploratory purposes and was not interpreted as evidence of causation.

---

## 5. Feature Engineering and Preprocessing

The features were divided into numerical and categorical groups.

### Numerical Features

```text
age
trestbps
chol
thalach
oldpeak
```

The numerical preprocessing pipeline performs:

1. Median missing-value imputation
2. Standard scaling

### Categorical Features

```text
sex
cp
fbs
restecg
exang
slope
ca
thal
```

The categorical preprocessing pipeline performs:

1. Most-frequent missing-value imputation
2. One-hot encoding

The preprocessing operations are implemented using Scikit-learn:

```text
Pipeline
ColumnTransformer
SimpleImputer
StandardScaler
OneHotEncoder
```

This ensures that preprocessing is reproducible and that the same transformations are applied during both training and inference.

---

## 6. Model Development

Two classification algorithms were evaluated:

1. Logistic Regression
2. Random Forest

The dataset was divided into training and testing datasets using an 80/20 split.

```text
Training samples: 242
Testing samples: 61
```

Stratified splitting was used to preserve the target class distribution.

---

## 7. Baseline Model Results

### Logistic Regression

| Metric | Score |
|---|---:|
| Accuracy | 0.8852 |
| Precision | 0.8387 |
| Recall | 0.9286 |
| ROC-AUC | 0.9665 |

### Random Forest

| Metric | Score |
|---|---:|
| Accuracy | 0.8525 |
| Precision | 0.8065 |
| Recall | 0.8929 |
| ROC-AUC | 0.9432 |

---

## 8. Cross-Validation

Five-fold cross-validation was performed on both models.

| Model | CV Accuracy | CV Precision | CV Recall | CV ROC-AUC |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.8428 | 0.8686 | 0.7739 | 0.8937 |
| Random Forest | 0.8055 | 0.8050 | 0.7648 | 0.8983 |

---

## 9. Hyperparameter Tuning

Hyperparameter tuning was performed using `GridSearchCV`.

### Logistic Regression

Best parameters:

```text
C = 0.1
class_weight = None
```

Best cross-validation ROC-AUC:

```text
0.8982
```

### Random Forest

Best parameters:

```text
n_estimators = 100
max_depth = 5
min_samples_split = 5
```

Best cross-validation ROC-AUC:

```text
0.9043
```

---

## 10. Final Model Evaluation

The tuned models produced the following test results:

| Model | Accuracy | Precision | Recall | ROC-AUC |
|---|---:|---:|---:|---:|
| Tuned Logistic Regression | 0.8852 | 0.8387 | 0.9286 | 0.9654 |
| Tuned Random Forest | 0.8525 | 0.8276 | 0.8571 | 0.9470 |

### Selected Model

**Tuned Logistic Regression** was selected as the final production model.

Its final test performance was:

```text
Accuracy  = 88.52%
Precision = 83.87%
Recall    = 92.86%
ROC-AUC   = 96.54%
```

---

## 11. MLflow Experiment Tracking

MLflow is used to track machine-learning experiments.

Experiment name:

```text
heart-disease-classification
```

Tracked runs include:

```text
Tuned_Logistic_Regression
Tuned_Random_Forest
```

MLflow records:

- Model type
- Hyperparameters
- Cross-validation configuration
- Accuracy
- Precision
- Recall
- ROC-AUC
- Best cross-validation ROC-AUC
- Model comparison artifact
- Trained model artifact

MLflow experiment screenshots are stored in:

```text
screenshots/
```

---

## 12. Model Packaging and Reproducibility

The final preprocessing pipeline and classifier are packaged together and serialized using Joblib.

The saved model is:

```text
models/heart_disease_pipeline.joblib
```

The reusable training script is:

```text
src/train.py
```

Running:

```bash
python src/train.py
```

recreates and saves the production model.

Project dependencies are documented in:

```text
requirements.txt
```

---

## 13. Automated Testing

Automated tests are implemented using `pytest`.

Tests cover:

- Processed dataset availability
- Binary target validation
- Expected dataset columns
- Saved model loading
- Model prediction behavior
- API functionality

Run the tests with:

```bash
pytest
```

The implemented tests successfully pass in the local environment and CI pipeline.

---

## 14. Code Quality

Ruff is used for Python linting and code-quality validation.

Run:

```bash
ruff check .
```

The same linting check is executed automatically by the CI pipeline.

---

## 15. CI/CD Pipeline

GitHub Actions is used to automate continuous integration.

Workflow configuration:

```text
.github/workflows/ci.yml
```

The CI pipeline runs automatically for pushes and pull requests to the `main` branch.

The workflow performs:

1. Repository checkout
2. Python environment setup
3. Dependency installation
4. Ruff linting
5. Pytest unit testing
6. Model training
7. Trained-model artifact upload

A successful GitHub Actions workflow demonstrates that the project can be reproduced in a clean environment.

---

## 16. FastAPI Model Serving

The trained model is exposed through a FastAPI application.

API source:

```text
api/main.py
```

Available endpoints include:

```text
GET  /
GET  /health
GET  /metrics
POST /predict
```

### Prediction Endpoint

The `/predict` endpoint accepts patient information as JSON.

Example request:

```json
{
  "age": 63,
  "sex": 1,
  "cp": 1,
  "trestbps": 145,
  "chol": 233,
  "fbs": 1,
  "restecg": 2,
  "thalach": 150,
  "exang": 0,
  "oldpeak": 2.3,
  "slope": 3,
  "ca": 0,
  "thal": 6
}
```

Example response:

```json
{
  "prediction": 0,
  "confidence": 0.5762
}
```

Prediction latency is also monitored by the application.

Interactive API documentation is available through:

```text
/docs
```

---

## 17. Docker Containerization

The FastAPI application is containerized using Docker.

The Docker configuration is defined in:

```text
Dockerfile
```

### Build the Image

```bash
docker build -t heart-disease-api .
```

### Run the Container

```bash
docker run -d \
  --name heart-disease-api-container \
  -p 8000:8000 \
  heart-disease-api
```

The API is then available on port:

```text
8000
```

---

## 18. Kubernetes Deployment

The containerized application is deployed locally using Minikube.

Kubernetes manifests are stored in:

```text
deployment/
```

Files include:

```text
deployment.yaml
service.yaml
```

The Kubernetes deployment uses:

- Two application replicas
- Container port 8000
- Readiness probe
- Liveness probe
- NodePort Service

Example deployment command:

```bash
minikube kubectl -- apply -f deployment/deployment.yaml
```

Example service deployment:

```bash
minikube kubectl -- apply -f deployment/service.yaml
```

The deployed API was verified using the `/health` and `/predict` endpoints.

---

## 19. Monitoring and Logging

Application monitoring is implemented using Prometheus.

Prometheus configuration:

```text
monitoring/prometheus.yml
```

FastAPI exposes metrics through:

```text
GET /metrics
```

Custom metrics include:

```text
api_requests_total
model_predictions_total
model_prediction_latency_seconds
```

These metrics allow monitoring of:

- API request volume
- Prediction distribution
- Prediction latency

Application logs also record prediction results, confidence, and latency.

---

## 20. MLOps Architecture

The overall workflow is:

```text
UCI Heart Disease Dataset
            |
            v
      Data Acquisition
            |
            v
       Data Cleaning
            |
            v
            EDA
            |
            v
 Feature Engineering
            |
            v
 Logistic Regression
    + Random Forest
            |
            v
  Cross-Validation
            |
            v
 Hyperparameter Tuning
            |
            v
          MLflow
            |
            v
 Final Logistic Regression
            |
            v
     Joblib Pipeline
            |
            v
     Pytest + Ruff
            |
            v
     GitHub Actions
            |
            v
         FastAPI
            |
            v
          Docker
            |
            v
 Kubernetes / Minikube
            |
            v
       Prometheus
```

---

## 21. Project Structure

```text
heart-disease-mlops/
|
├── .github/
│   └── workflows/
│       └── ci.yml
|
├── api/
│   ├── __init__.py
│   └── main.py
|
├── data/
│   ├── raw/
│   └── processed/
|
├── deployment/
│   ├── deployment.yaml
│   └── service.yaml
|
├── models/
│   └── heart_disease_pipeline.joblib
|
├── monitoring/
│   └── prometheus.yml
|
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_model_experiments.ipynb
|
├── screenshots/
|
├── src/
│   ├── download_data.py
│   ├── data_processing.py
│   ├── features.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
|
├── tests/
│   ├── test_api.py
│   ├── test_data_processing.py
│   └── test_features.py
|
├── .gitignore
├── Dockerfile
├── README.md
└── requirements.txt
```

---

## 22. Reproducing the Project

Clone the repository:

```bash
git clone https://github.com/Gudiya-1008-pilani/heart-disease-mlops.git
```

Enter the project:

```bash
cd heart-disease-mlops
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Download the dataset:

```bash
python src/download_data.py
```

Train the model:

```bash
python src/train.py
```

Run tests:

```bash
pytest
```

Run linting:

```bash
ruff check .
```

Run the API:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Open Swagger documentation at:

```text
http://localhost:8000/docs
```

---

## 23. Challenges and Solutions

Several practical MLOps challenges were encountered during implementation.

### Environment Reproducibility

Dependencies from the development VM initially caused CI installation problems.

This was resolved by creating a clean `requirements.txt` containing reproducible package versions.

### MLflow Tracking Location

MLflow experiment data was initially written to different SQLite databases depending on the working directory.

The correct tracking database was explicitly configured to ensure the experiment runs were visible in the MLflow dashboard.

### CI/CD Debugging

Initial GitHub Actions workflow runs failed because of environment-specific dependencies.

After cleaning the dependency configuration, the workflow successfully completed linting, tests, model training, and artifact upload.

### Docker Deployment

The initial Dockerfile was empty, preventing image creation.

A reproducible Python-based Docker image was subsequently configured and successfully used to run the FastAPI service.

### Kubernetes Tooling

The environment did not provide a standalone `kubectl` command.

Minikube's bundled kubectl interface was therefore used through:

```bash
minikube kubectl --
```

---

## 24. Future Improvements

Possible future improvements include:

- Larger and more diverse training datasets
- Automated model retraining
- Data drift detection
- Model performance drift detection
- MLflow Model Registry integration
- Automated model approval workflows
- Grafana dashboards
- Cloud-managed Kubernetes deployment
- Infrastructure as Code
- Automated rollback mechanisms
- Secrets management
- Authentication and authorization for the prediction API
- HTTPS/TLS support
- Automated vulnerability scanning

---

## 25. Conclusion

This project demonstrates a complete end-to-end MLOps workflow for a binary heart disease classification problem.

The workflow covers the full lifecycle from data acquisition and model experimentation through reproducible packaging, automated testing, CI/CD, API serving, Docker containerization, Kubernetes deployment, and production-style monitoring.

Among the evaluated models, tuned Logistic Regression provided the strongest final test performance, achieving an accuracy of 88.52%, recall of 92.86%, and ROC-AUC of 96.54%.

The project demonstrates how modern MLOps practices can make machine-learning systems reproducible, testable, deployable, and observable.