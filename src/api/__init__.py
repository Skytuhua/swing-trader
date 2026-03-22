"""
SwingTrader Admin API package.

This package contains the FastAPI application factory, middleware, Pydantic schemas,
and all route handlers for the administrative dashboard API.
"""

from src.api.app import create_app

__all__ = ["create_app"]
