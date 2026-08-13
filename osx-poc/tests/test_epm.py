"""Test M3 — EPM: Expert Position Memory I/O layer (issue #27).

Pura logica di file system (snapshot + storico run) — nessuna dipendenza
da EAT/GCSGWorker/vLLM, vedi src/scheduler/epm.py per il perché è
separata da entrambi.
"""
import json

from scheduler import epm


# ── snapshot ─────────────────────────────────────────────────────────────────

class TestSnapshotFile:

    def test_load_missing_file_returns_none(self, tmp_path):
        assert epm.load_snapshot_file(tmp_path / "missing.json") is None

    def test_load_corrupt_file_returns_none(self, tmp_path):
        path = tmp_path / "corrupt.json"
        path.write_text("{not valid json")
        assert epm.load_snapshot_file(path) is None

    def test_write_then_load_roundtrip(self, tmp_path):
        path = tmp_path / "nested" / "snapshot.json"
        snapshot = {"version": 1, "exported_at": 123.0, "entries": {"0:0": {"access_count": 5}}}

        epm.write_snapshot_file(snapshot, path)
        loaded = epm.load_snapshot_file(path)

        assert loaded == snapshot

    def test_write_creates_parent_directories(self, tmp_path):
        path = tmp_path / "a" / "b" / "c" / "snapshot.json"
        epm.write_snapshot_file({"version": 1, "entries": {}}, path)
        assert path.exists()

    def test_write_no_leftover_tmp_file(self, tmp_path):
        path = tmp_path / "snapshot.json"
        epm.write_snapshot_file({"version": 1, "entries": {}}, path)
        assert not path.with_suffix(".json.tmp").exists()


# ── storico run ──────────────────────────────────────────────────────────────

class TestRunHistory:

    def test_load_missing_file_returns_empty_list(self, tmp_path):
        assert epm.load_history(tmp_path / "missing.json") == []

    def test_load_corrupt_file_returns_empty_list(self, tmp_path):
        path = tmp_path / "corrupt.json"
        path.write_text("not json at all")
        assert epm.load_history(path) == []

    def test_load_non_list_json_returns_empty_list(self, tmp_path):
        """Difensivo: un file che contiene un oggetto invece di una lista
        (es. scritto a mano/da un altro tool) non deve far esplodere il
        chiamante — vedi lo stesso principio in load_snapshot_file()."""
        path = tmp_path / "not_a_list.json"
        path.write_text(json.dumps({"oops": "not a list"}))
        assert epm.load_history(path) == []

    def test_append_single_record(self, tmp_path):
        path = tmp_path / "history.json"
        epm.append_run_record({"run_id": "r1"}, path)
        assert epm.load_history(path) == [{"run_id": "r1"}]

    def test_append_preserves_order(self, tmp_path):
        path = tmp_path / "history.json"
        epm.append_run_record({"run_id": "r1"}, path)
        epm.append_run_record({"run_id": "r2"}, path)
        epm.append_run_record({"run_id": "r3"}, path)

        history = epm.load_history(path)

        assert [r["run_id"] for r in history] == ["r1", "r2", "r3"]

    def test_append_caps_at_max_runs_fifo(self, tmp_path):
        """Profondità limitata (richiesta esplicita, issue #27): lo storico
        non cresce senza limite — oltre max_runs si perde il più vecchio,
        non l'ultimo appena scritto."""
        path = tmp_path / "history.json"
        for i in range(5):
            epm.append_run_record({"run_id": f"r{i}"}, path, max_runs=3)

        history = epm.load_history(path)

        assert [r["run_id"] for r in history] == ["r2", "r3", "r4"]

    def test_default_max_history_runs_is_256(self):
        assert epm.MAX_HISTORY_RUNS == 256

    def test_append_respects_default_cap(self, tmp_path):
        path = tmp_path / "history.json"
        for i in range(260):
            epm.append_run_record({"run_id": f"r{i}"}, path)

        history = epm.load_history(path)

        assert len(history) == 256
        assert history[0]["run_id"] == "r4"     # i più vecchi (r0..r3) sono caduti
        assert history[-1]["run_id"] == "r259"


# ── positions_match ──────────────────────────────────────────────────────────

class TestPositionsMatch:

    def test_matching_sets_regardless_of_order(self):
        assert epm.positions_match([3, 1, 2], [1, 2, 3]) is True

    def test_mismatched_sets(self):
        assert epm.positions_match([1, 2], [1, 3]) is False

    def test_none_prev_final(self):
        assert epm.positions_match(None, [1, 2]) is False

    def test_none_next_initial(self):
        assert epm.positions_match([1, 2], None) is False

    def test_both_none(self):
        assert epm.positions_match(None, None) is False

    def test_both_empty_lists_match(self):
        """Due pool vuoti (es. shadow_pool_size=0) contano come 'coincidono' —
        non un caso interessante, ma non un falso negativo nemmeno."""
        assert epm.positions_match([], []) is True


# ── new_run_id ───────────────────────────────────────────────────────────────

class TestNewRunId:

    def test_format_has_epoch_and_hex_suffix(self):
        run_id = epm.new_run_id()
        epoch_part, hex_part = run_id.split("-")
        assert epoch_part.isdigit()
        assert len(hex_part) == 8
        int(hex_part, 16)   # non deve sollevare — è esadecimale valido

    def test_unique_across_calls(self):
        ids = {epm.new_run_id() for _ in range(100)}
        assert len(ids) == 100
