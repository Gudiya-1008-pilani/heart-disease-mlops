# Heart Disease MLOps — Deployment Access Instructions

## Deployment Environment

The Heart Disease Prediction API is deployed locally using Docker and Kubernetes (Minikube).

The application exposes a REST API through FastAPI.

## Prerequisites

- Python 3.12
- Docker
- Minikube
- Kubernetes CLI (or Minikube's bundled kubectl)

## Deployment Steps

**1. Start Minikube**

```bash
minikube start --driver=docker
```

**2. Build the Docker image inside Minikube**

```bash
eval $(minikube docker-env)
docker build -t heart-disease-api:latest .
```

**3. Deploy the application**

```bash
minikube kubectl -- apply -f deployment/deployment.yaml
minikube kubectl -- apply -f deployment/service.yaml
```

**4. Check the deployment**

```bash
minikube kubectl -- get deployments
minikube kubectl -- get pods
minikube kubectl -- get services
```

The expected configuration is two application replicas.

**5. Expose the application**

```bash
minikube kubectl -- port-forward service/heart-disease-api-service 8080:8000
```

Keep this terminal running.

## API Access

From the same environment, use:

- API documentation: `http://localhost:8080/docs`
- Health endpoint: `http://localhost:8080/health`
- Prediction endpoint: `http://localhost:8080/predict`

If accessing a remote VM through VS Code, forward port `8080` using the Ports panel.

## Sample Prediction Request

```bash
curl -X POST "http://localhost:8080/predict" \
-H "Content-Type: application/json" \
-d '{"age":63,"sex":1,"cp":1,"trestbps":145,"chol":233,"fbs":1,"restecg":2,"thalach":150,"exang":0,"oldpeak":2.3,"slope":3,"ca":0,"thal":6}'
```

The API returns a binary prediction and a confidence score. The monitored version also returns inference latency.

## Monitoring

The FastAPI application exposes Prometheus metrics at `/metrics`.

The Prometheus configuration is located at:

`monitoring/prometheus.yml`

Monitoring metrics include API request count, prediction distribution, and prediction latency.

## Note

This is a local Kubernetes deployment, not a publicly hosted API. Minikube and port-forwarding must be running to access the service.
