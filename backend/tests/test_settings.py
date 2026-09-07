"""Tests for environment-derived application settings."""

import unittest
from unittest.mock import patch

from app.config import Settings


class TestSettingsDatabaseUrl(unittest.TestCase):
    def test_database_url_is_built_from_postgres_fields_when_blank(self) -> None:
        settings = Settings(
            _env_file=None,
            encryption_key="0" * 64,
            secret_key="t" * 32,
            database_url="",
            postgres_host="db",
            postgres_port=5432,
            postgres_user="heym",
            postgres_password="secret",
            postgres_db="workflow",
        )

        self.assertEqual(
            settings.database_url,
            "postgresql+asyncpg://heym:secret@db:5432/workflow",
        )

    def test_explicit_database_url_takes_precedence(self) -> None:
        settings = Settings(
            _env_file=None,
            encryption_key="0" * 64,
            secret_key="t" * 32,
            database_url="postgresql+asyncpg://explicit:secret@custom:6544/app",
            postgres_host="db",
            postgres_port=5432,
            postgres_user="heym",
            postgres_password="secret",
            postgres_db="workflow",
        )

        self.assertEqual(
            settings.database_url,
            "postgresql+asyncpg://explicit:secret@custom:6544/app",
        )


class TestSettingsVersion(unittest.TestCase):
    def test_resolved_version_reads_version_file_when_app_version_is_blank(self) -> None:
        settings = Settings(
            _env_file=None, encryption_key="0" * 64, secret_key="t" * 32, app_version=""
        )

        with patch("app.config._read_version", return_value="0.0.24"):
            self.assertEqual(settings.resolved_version, "0.0.24")

    def test_explicit_app_version_takes_precedence(self) -> None:
        settings = Settings(
            _env_file=None, encryption_key="0" * 64, secret_key="t" * 32, app_version="0.0.25"
        )

        with patch("app.config._read_version", return_value="0.0.24"):
            self.assertEqual(settings.resolved_version, "0.0.25")
