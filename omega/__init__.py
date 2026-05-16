"""Omega package bootstrap.

Load a local .env file for developer convenience if python-dotenv is
installed. Production deployments should supply environment variables
through a secure mechanism and not rely on .env files.
"""
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    # If python-dotenv is not installed, do nothing — env vars must be set
    # by the runtime environment in CI/production.
    pass
