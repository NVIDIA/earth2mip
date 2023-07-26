from pathlib import Path
from earth2mip import filesystem


def test_glob(tmp_path: Path):
    a = tmp_path / "a.txt"
    a.touch()

    # use file:// protocol to ensure handling is correct
    (f,) = filesystem.glob(f"file://{tmp_path.as_posix()}/*.txt")
    assert f == f"file://{a.as_posix()}"


def test_glob_no_scheme(tmp_path: Path):
    a = tmp_path / "a.txt"
    a.touch()

    (f,) = filesystem.glob(f"{tmp_path.as_posix()}/*.txt")
    assert f == a.as_posix()


def test__to_url():
    assert (
        filesystem._to_url("s3", "sw_climate_fno/a.txt") == "s3://sw_climate_fno/a.txt"
    )
    assert filesystem._to_url("", "sw_climate_fno/a.txt") == "sw_climate_fno/a.txt"
