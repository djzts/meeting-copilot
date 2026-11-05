import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture
def vault_path() -> Path:
    return FIXTURES / "vault"


@pytest.fixture
def vault(vault_path, tmp_path):
    from vault import Vault

    v = Vault(str(vault_path), exclude=["NDA/**"], embedding_model="", cache_dir=str(tmp_path))
    v.build()
    return v
