import sys
from earth2mip.networks import depends_on_time

if sys.version_info < (3, 10):
    from importlib_metadata import EntryPoint
else:
    from importlib.metadata import EntryPoint

import pytest


def test_upstream_entrypoint():
    ep = EntryPoint(name=None, group=None, value="sys")
    assert ep.load() == sys

    # refer to an attribute with ":"
    ep = EntryPoint(name=None, group=None, value="sys:argv")
    assert ep.load() == sys.argv

    # if you don't use : it will give an error
    with pytest.raises(ModuleNotFoundError):
        ep = EntryPoint(name=None, group=None, value="sys.argv")
        ep.load()


def test_inspect_for_time():
    def f(x, time):
        pass

    def g(x):
        pass

    assert depends_on_time(f)
    assert not depends_on_time(g)
