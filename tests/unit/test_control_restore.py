from mm.runtime.control_restore import boot_paused_actors


def test_boot_paused_actors_pauses_everyone_without_snapshot() -> None:
    assert boot_paused_actors({"a", "b"}, None) == {"a", "b"}


def test_boot_paused_actors_restores_last_pause_set() -> None:
    snapshot = {
        "type": "RuntimeControlState",
        "actors": ["a", "b"],
        "paused_actors": ["a"],
    }
    assert boot_paused_actors({"a", "b"}, snapshot) == {"a"}


def test_boot_paused_actors_pauses_new_actors_only() -> None:
    snapshot = {
        "type": "RuntimeControlState",
        "actors": ["a"],
        "paused_actors": [],
    }
    assert boot_paused_actors({"a", "b"}, snapshot) == {"b"}
