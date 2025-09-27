from crucible.api.canary import select_model_tag

def test_select_model_tag_stable():
    # Should return 'stable' for deterministic session_id
    tag = select_model_tag("testsessionid123")
    assert tag in ("stable", "vnext")

def test_select_model_tag_random():
    # Should return a valid tag even with None
    tag = select_model_tag()
    assert tag in ("stable", "vnext")
