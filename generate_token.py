"""
One-time-per-day Fyers login: generates an access_token and saves it locally
to .env (which is NOT delivered/tracked -- it stays only on your machine).

Usage:
    python generate_token.py

You'll be prompted to open a URL, log in to Fyers in your browser, and paste
back the redirected URL (or just the auth_code from it). Your credentials
(APP_ID, SECRET_KEY, REDIRECT_URI) are read from .env -- copy .env.example to
.env and fill them in first; this script never asks you to paste your secret
into chat or a shared file again.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv, set_key

ENV_PATH = Path(__file__).parent / ".env"

if not ENV_PATH.exists():
    print("No .env file found. Copy .env.example to .env and fill in APP_ID, "
          "SECRET_KEY and REDIRECT_URI first.")
    sys.exit(1)

load_dotenv(ENV_PATH)

APP_ID = os.getenv("FYERS_APP_ID")
SECRET_KEY = os.getenv("FYERS_SECRET_KEY")
REDIRECT_URI = os.getenv("FYERS_REDIRECT_URI")

missing = [n for n, v in [("FYERS_APP_ID", APP_ID), ("FYERS_SECRET_KEY", SECRET_KEY),
                           ("FYERS_REDIRECT_URI", REDIRECT_URI)] if not v]
if missing:
    print(f"Missing from .env: {', '.join(missing)}")
    sys.exit(1)

try:
    from fyers_apiv3 import fyersModel
except ImportError:
    print("fyers-apiv3 is not installed. Run: pip install fyers-apiv3")
    sys.exit(1)

session = fyersModel.SessionModel(
    client_id=APP_ID,
    secret_key=SECRET_KEY,
    redirect_uri=REDIRECT_URI,
    response_type="code",
    grant_type="authorization_code",
)

auth_url = session.generate_authcode()
print("\n1. Open this URL, log in to Fyers, and approve access:\n")
print(auth_url)
print("\n2. You'll be redirected to a URL that doesn't load (that's expected "
      "since the redirect URL isn't a real server) -- copy the FULL URL from "
      "your browser's address bar and paste it below.\n")

pasted = input("Paste the redirected URL (or just the auth_code): ").strip()

match = re.search(r"auth_code=([^&]+)", pasted)
auth_code = match.group(1) if match else pasted

session.set_token(auth_code)
response = session.generate_token()

if "access_token" not in response:
    print(f"Token generation failed: {response}")
    sys.exit(1)

access_token = response["access_token"]
set_key(str(ENV_PATH), "FYERS_ACCESS_TOKEN", access_token)
print(f"\nAccess token saved to {ENV_PATH} (valid for today).")
print("You can now run: streamlit run app.py")
