"""Test-run defaults that must not come from the developer's own .env.

`Settings` reads .env at import, so a machine that runs Heym with a cluster
enabled ran different code than CI: API, trigger and cron runs were enqueued
onto the run queue instead of executed in-process, and a dozen suites failed
here and nowhere else. Pinning the cluster off makes a local run mean what a
CI run means; the suites that are about the cluster turn it back on themselves.
"""

from unittest.mock import patch

import pytest

from app.config import settings


@pytest.fixture(autouse=True)
def cluster_disabled_unless_a_test_asks_for_it():
    with patch.object(settings, "cluster_enabled", False):
        yield
