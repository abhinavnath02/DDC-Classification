from fastapi.testclient import TestClient
from pathlib import Path

import sys
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.api import app

client = TestClient(app)

def test_frontend_loads():
    """Test that the static frontend loads correctly."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "DDC Book Classifier — Dewey Decimal Classification" in response.text
