from earth2mip import registry


def test_list_models(has_registry):
    ans = registry.list_models()
    assert ans
    assert "/" not in ans[0], ans[0]
