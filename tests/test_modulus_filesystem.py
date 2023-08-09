import pytest
from pathlib import Path
from modulus.utils import filesystem


@pytest.fixture
def pyfile_name():
    return "test_filesystem.py"


def test_modulus_filesystem_local(pyfile_name):
    # Check if this test file is seen in a Fsspec local file system
    file_path = Path(__file__).parent.resolve()
    fssystem = filesystem._get_fs("file")
    assert pyfile_name in [Path(file).name for file in fssystem.ls(file_path)]
