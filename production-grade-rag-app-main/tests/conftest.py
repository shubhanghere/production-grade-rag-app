import os

# Settings() runs at import time and requires these — set dummies so the test
# suite never depends on real credentials or a populated .env.
os.environ.setdefault("ANTHROPIC_API_KEY", "test-anthropic-key")
os.environ.setdefault("VOYAGE_API_KEY", "test-voyage-key")
