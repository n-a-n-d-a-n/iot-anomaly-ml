"""Entry point for the IoT Anomaly Detection API server."""

from __future__ import annotations

import uvicorn


def main() -> None:
    """Run the FastAPI server on port 8000."""
    uvicorn.run("src.api.routes:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
