import os
import dagshub
import mlflow

from src.logger.logger import get_logger

logger = get_logger(__name__)

_initialized = False


def init_dagshub() -> None:
    """
    Initialize DAGsHub as the remote MLflow tracking backend.
    Reads repo owner/name from environment variables so it works
    identically in local dev, Docker, and CI/CD pipelines.

    Required environment variables:
        DAGSHUB_USER_TOKEN  – Personal access token from DAGsHub settings
        DAGSHUB_REPO_OWNER  – DAGsHub username (default: muhammaduzair1411)
        DAGSHUB_REPO_NAME   – DAGsHub repository name (default: my-first-repo)
    """
    global _initialized  # noqa: PLW0603
    if _initialized:
        return

    repo_owner = os.getenv("DAGSHUB_REPO_OWNER", "muhammaduzair1411")
    repo_name  = os.getenv("DAGSHUB_REPO_NAME",  "my-first-repo")

    dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)

    tracking_uri = mlflow.get_tracking_uri()
    logger.info(f"DAGsHub initialized — tracking URI: {tracking_uri}")

    _initialized = True
