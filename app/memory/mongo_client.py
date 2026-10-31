"""MongoDB Atlas connection manager for Adaptive MAS."""

import logging
import sys
from typing import Optional
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.errors import (
    AutoReconnect,
    ConnectionFailure,
    NetworkTimeout,
    PyMongoError,
    ServerSelectionTimeoutError,
)
from app.config.settings import settings

logger = logging.getLogger(__name__)

_mongo_client: Optional[MongoClient] = None
_mongo_init_failed: bool = False


def reset_mongo_client() -> None:
    """Reset the MongoDB client singleton (primarily for testing)."""
    global _mongo_client, _mongo_init_failed
    _mongo_client = None
    _mongo_init_failed = False


def check_mongo_network_error(exc: Exception) -> None:
    """Handle network or disconnection errors during runtime operations."""
    if isinstance(
        exc,
        (
            ConnectionFailure,
            ServerSelectionTimeoutError,
            AutoReconnect,
            NetworkTimeout,
            PyMongoError,
        ),
    ):
        status_msg = "Status: Please check your internet connection."
        print(f"\n{status_msg}")
        raise ConnectionError(status_msg) from exc
    raise exc


def get_mongo_client(required: bool = False) -> Optional[MongoClient]:
    """Get or initialize the MongoDB client singleton.

    If required=True, prints an error and terminates process if MongoDB
    cannot be configured or connected.
    """
    global _mongo_client, _mongo_init_failed
    if _mongo_client is not None:
        return _mongo_client

    if _mongo_init_failed and not required:
        return None

    if not settings.mongodb_uri:
        msg = "Unable to configure MongoDB: MONGODB_URI is not set."
        print(f"\n{msg}")
        _mongo_init_failed = True
        if required:
            sys.exit(1)
        return None

    try:
        _mongo_client = MongoClient(
            settings.mongodb_uri,
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=5000,
        )
        # Verify connection and credentials
        _mongo_client.admin.command("ping")
        logger.info("Connected successfully to MongoDB Atlas")
        _mongo_init_failed = False
        return _mongo_client
    except Exception as e:
        msg = f"Unable to configure MongoDB: {e}"
        print(f"\n{msg}")
        _mongo_client = None
        _mongo_init_failed = True
        if required:
            sys.exit(1)
        return None


def get_mongo_db(required: bool = False) -> Optional[Database]:
    """Get the active MongoDB database instance."""
    client = get_mongo_client(required=required)
    if client is not None:
        try:
            default_db = client.get_default_database()
            if default_db is not None:
                return default_db
        except Exception:
            pass
        return client[settings.mongodb_db_name]
    return None
