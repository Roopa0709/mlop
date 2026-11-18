# California Housing Price Prediction

Run a four-configuration experiment on the California Housing dataset. It
compares StandardScaler and MinMaxScaler preprocessing with Ridge regression
at two regularization strengths (`alpha=0.1` and `alpha=10`). The target is
expressed in units of $100,000. Each run logs its parameters, MAE, R-squared,
and a model with its input signature and example to the local MLflow store.

## Run training

From PowerShell in this directory:

```powershell
.\myenv\Scripts\python.exe .\california_housing_mlflow.py
```

The script uses `mlflow.db` for tracking and stores model artifacts under
`mlruns\california-housing-price-prediction`.
After all four runs complete, it selects the model with the highest R-squared
(using lower MAE to break ties), registers that run's model as
`california-housing-price-predictor`, and sets its `champion` alias to the new
version. Each execution creates a new registry version and updates the alias
to the best configuration from that execution.

## Explore MLflow

Start the tracking UI in a separate PowerShell window:

```powershell
.\myenv\Scripts\mlflow.exe ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root .\mlruns --host 127.0.0.1 --port 5000
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000), choose the
`california-housing-price-prediction` experiment. Filter runs using
`tags.comparison_group = 'ridge-scaler-grid'` to focus on the four new
configurations, then compare `mae` (lower is better) and `r2` (higher is
better). All four runs use the same 20% holdout split and random seed, making
the preprocessing and `alpha` settings straightforward to compare. Open a run
to inspect its logged parameters and model artifact. For the selected model,
open **Models** in MLflow and inspect the `california-housing-price-predictor`
registered model and its `champion` alias. Previous baseline runs remain in
the experiment for reference.

Load the current champion for inference from this directory:

```python
import mlflow
from sklearn.datasets import fetch_california_housing

mlflow.set_tracking_uri("sqlite:///mlflow.db")
model = mlflow.pyfunc.load_model(
    "models:/california-housing-price-predictor@champion"
)
sample = fetch_california_housing(as_frame=True).data.iloc[:1]
print(model.predict(sample))
```

## Serve predictions with Flask

First run the training script above so the `champion` model alias exists. Then
install the project dependencies and start the API from this directory:

```powershell
.\myenv\Scripts\python.exe -m pip install -r requirements.txt
.\myenv\Scripts\python.exe .\app.py
```

The API loads the current MLflow champion once at startup and listens at
`http://127.0.0.1:8000`. Check `GET http://127.0.0.1:8000/health` to confirm
it is running.

To try a prediction in Postman, send a `POST` request to
`http://127.0.0.1:8000/predict`, select **Body > raw > JSON**, and use this
body. All eight feature names and numeric values are required:

```json
{
  "features": {
    "MedInc": 8.3252,
    "HouseAge": 41.0,
    "AveRooms": 6.9841,
    "AveBedrms": 1.0238,
    "Population": 322,
    "AveOccup": 2.5556,
    "Latitude": 37.88,
    "Longitude": -122.23
  }
}
```

A successful response contains the predicted target (in units of $100,000)
and the equivalent estimated price in dollars. Invalid JSON, missing or
unexpected feature names, and non-numeric or non-finite values return HTTP
400 with a JSON error.

For a GET request, provide the same feature names as query parameters in
Postman with no request body:

```text
http://127.0.0.1:8000/predict?MedInc=8.3252&HouseAge=41.0&AveRooms=6.9841&AveBedrms=1.0238&Population=322&AveOccup=2.5556&Latitude=37.88&Longitude=-122.23
```

GET is convenient for quick tests, but query parameters can be recorded in
browser or proxy logs. Prefer POST for normal use.

## Streamlit web interface and Docker

Export the current champion model to a portable local bundle, then start the
Streamlit interface:

```powershell
.\myenv\Scripts\python.exe -m pip install -r requirements.txt
.\myenv\Scripts\python.exe .\export_model.py
.\myenv\Scripts\python.exe -m streamlit run .\streamlit_app.py
```

Open [http://localhost:8501](http://localhost:8501), enter the area details,
and choose **Estimate house value**. The result is displayed in US dollars.
The model bundle is stored in `model\`. If you train and register a new
champion, remove the old `model\` bundle and run `export_model.py` again.

Build and run the container from the project directory:

Ensure Docker Desktop is running with the Linux container engine enabled.

```powershell
docker build -t california-housing-ui .
docker run --rm -p 8501:8501 california-housing-ui
```

Then open [http://localhost:8501](http://localhost:8501). The Docker image
contains the Streamlit app and exported model; it does not need the host's
MLflow database or `mlruns\` directory at runtime.

## GitHub Actions validation

The workflow at `.github/workflows/docker.yml` runs on pushes, pull requests,
and manual dispatch. It builds the Docker image, checks that the Streamlit app
responds, and runs a prediction using the model inside the container. After
these checks pass, pushes publish a commit-specific image to Docker Hub; pushes
to the default branch also update the `latest` tag. Pull requests only build
and test, and do not publish.

To enable publishing:

1. Create a Docker Hub repository named `california-housing-ui`.
2. Create a Docker Hub access token with read/write permission.
3. In the GitHub repository, open **Settings > Secrets and variables >
   Actions > New repository secret** and add `DOCKERHUB_USERNAME` with your
   Docker Hub username and `DOCKERHUB_TOKEN` with the access token.
4. Push this project to GitHub with the `roopa` folder as the repository root.
   The workflow will run automatically. Check the **Actions** tab for build,
   validation, and publishing results.

Every successful push creates a tag such as
`<dockerhub-username>/california-housing-ui:sha-<commit-sha>`. On the default
branch it also updates `<dockerhub-username>/california-housing-ui:latest`.

To download and run the latest default-branch image, replace
`YOUR_DOCKERHUB_USERNAME` with your Docker Hub username:

```powershell
docker pull YOUR_DOCKERHUB_USERNAME/california-housing-ui:latest
docker run --rm -p 8502:8501 YOUR_DOCKERHUB_USERNAME/california-housing-ui:latest
```

Open [http://localhost:8502](http://localhost:8502). The host port is `8502`
because `8501` may already be used by another local container.
