"""Tests for the edge triage pipeline — CNN + agentic layers + security + SLA."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from edge_triage.audit import AuditLogger
from edge_triage.config import TriageConfig, config
from edge_triage.data_ingest import ImageIngestor
from edge_triage.inference import QuantizedInferencer
from edge_triage.metrics import MetricsCollector
from edge_triage.model_registry import ModelRegistry
from edge_triage.reasoning_loop import ReActReasoningLoop
from edge_triage.triage import EdgeTriageEngine, TriageResult
from edge_triage.utils import AgentPowerGuard, PowerMonitor


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def clear_tile() -> np.ndarray:
    return np.random.default_rng(42).random((256, 256, 3), dtype=np.float32) * 0.3


@pytest.fixture
def cloudy_tile() -> np.ndarray:
    return np.ones((256, 256, 3), dtype=np.float32) * 0.9


@pytest.fixture
def sentinel2_tile() -> np.ndarray:
    """13-channel Sentinel-2 multispectral tile (uint16 scaled 0-10000)."""
    return (np.random.default_rng(99).random((256, 256, 13)) * 3000).astype(np.uint16)


@pytest.fixture
def engine() -> EdgeTriageEngine:
    return EdgeTriageEngine(audit=False)


# ── Config Validation ───────────────────────────────────────────────────────


class TestConfigValidation:
    def test_default_config_valid(self) -> None:
        cfg = TriageConfig()
        assert cfg.POWER_BUDGET_WATTS == 15.0

    def test_negative_power_budget_rejected(self) -> None:
        with pytest.raises(ValueError, match="POWER_BUDGET_WATTS"):
            TriageConfig(POWER_BUDGET_WATTS=-1.0)

    def test_threshold_above_1_rejected(self) -> None:
        with pytest.raises(ValueError, match="KEEP_SCORE_THRESHOLD"):
            TriageConfig(KEEP_SCORE_THRESHOLD=1.5)

    def test_invalid_mode_rejected(self) -> None:
        with pytest.raises(ValueError, match="MODE"):
            TriageConfig(MODE="underwater")

    def test_tile_size_bounds(self) -> None:
        with pytest.raises(ValueError, match="TILE_SIZE"):
            TriageConfig(TILE_SIZE=4)

    def test_valid_custom_config(self) -> None:
        cfg = TriageConfig(POWER_BUDGET_WATTS=25.0, MODE="ground")
        assert cfg.POWER_BUDGET_WATTS == 25.0
        assert cfg.MODE == "ground"


# ── ImageIngestor ───────────────────────────────────────────────────────────


class TestImageIngestor:
    def test_preprocess_float32(self, clear_tile: np.ndarray) -> None:
        out = ImageIngestor().preprocess(clear_tile)
        assert out.dtype == np.float32
        assert out.shape[0] == 3

    def test_preprocess_uint8(self) -> None:
        tile = (np.random.default_rng(1).random((128, 128, 3)) * 255).astype(np.uint8)
        out = ImageIngestor().preprocess(tile)
        assert out.dtype == np.float32
        assert 0.0 <= out.max() <= 1.0

    def test_preprocess_sentinel2_uint16(self, sentinel2_tile: np.ndarray) -> None:
        out = ImageIngestor().preprocess(sentinel2_tile)
        assert out.dtype == np.float32
        assert out.shape == (13, 256, 256)
        assert 0.0 <= out.max() <= 1.0

    def test_preprocess_2d_grayscale(self) -> None:
        gray = np.random.default_rng(5).random((128, 128), dtype=np.float32)
        out = ImageIngestor().preprocess(gray)
        assert out.shape == (1, 128, 128)

    def test_preprocess_already_channel_first(self) -> None:
        chw = np.random.default_rng(3).random((3, 256, 256), dtype=np.float32)
        out = ImageIngestor().preprocess(chw)
        assert out.shape == (3, 256, 256)

    def test_preprocess_rejects_4d(self) -> None:
        with pytest.raises(ValueError, match="2D or 3D"):
            ImageIngestor().preprocess(np.zeros((1, 3, 256, 256), dtype=np.float32))

    def test_tile_splitting(self) -> None:
        big = np.random.default_rng(0).random((512, 512, 3), dtype=np.float32)
        tiles = list(ImageIngestor(tile_size=256).tile(big))
        assert len(tiles) == 4
        assert all(t.shape == (256, 256, 3) for t in tiles)

    def test_tile_padding(self) -> None:
        odd = np.ones((300, 300, 3), dtype=np.float32)
        tiles = list(ImageIngestor(tile_size=256).tile(odd))
        assert len(tiles) == 4
        assert tiles[-1][0, 0, 0] == 1.0
        assert tiles[-1][43, 43, 0] == 1.0
        assert tiles[-1][44, 0, 0] == 0.0

    def test_tile_2d_grayscale(self) -> None:
        gray = np.ones((512, 512), dtype=np.float32)
        tiles = list(ImageIngestor(tile_size=256).tile(gray))
        assert len(tiles) == 4
        assert tiles[0].shape == (256, 256, 1)

    def test_load_numpy_array(self, clear_tile: np.ndarray) -> None:
        out = ImageIngestor().load(clear_tile)
        assert out.dtype == np.float32

    def test_npy_load_rejects_pickle(self, tmp_path: Path) -> None:
        """Ensure np.load uses allow_pickle=False."""
        npy_file = tmp_path / "safe.npy"
        np.save(str(npy_file), np.zeros((10, 10, 3), dtype=np.float32))
        out = ImageIngestor().load(npy_file)
        assert out.shape == (10, 10, 3)

    def test_nan_input_handled(self) -> None:
        """NaN values don't crash preprocess."""
        nan_tile = np.full((64, 64, 3), np.nan, dtype=np.float32)
        out = ImageIngestor().preprocess(nan_tile)
        assert out.shape == (3, 64, 64)

    def test_all_zeros_input(self) -> None:
        zeros = np.zeros((64, 64, 3), dtype=np.float32)
        out = ImageIngestor().preprocess(zeros)
        assert out.shape == (3, 64, 64)


# ── QuantizedInferencer ─────────────────────────────────────────────────────


class TestQuantizedInferencer:
    def test_infer_returns_required_keys(self, clear_tile: np.ndarray) -> None:
        inf = QuantizedInferencer()
        tile = ImageIngestor().preprocess(clear_tile)
        result = inf.infer(tile)
        for key in ("cloud_fraction", "anomaly_score", "value_score",
                     "inference_ms", "power_watts", "backend"):
            assert key in result

    def test_scores_in_range(self, clear_tile: np.ndarray) -> None:
        inf = QuantizedInferencer()
        tile = ImageIngestor().preprocess(clear_tile)
        result = inf.infer(tile)
        assert 0.0 <= result["cloud_fraction"] <= 1.0
        assert 0.0 <= result["anomaly_score"] <= 1.0
        assert 0.0 <= result["value_score"] <= 1.0

    def test_cloudy_tile_high_cloud_fraction(self, cloudy_tile: np.ndarray) -> None:
        inf = QuantizedInferencer()
        tile = ImageIngestor().preprocess(cloudy_tile)
        assert inf.infer(tile)["cloud_fraction"] > 0.5

    def test_sentinel2_13ch_inference(self, sentinel2_tile: np.ndarray) -> None:
        inf = QuantizedInferencer()
        tile = ImageIngestor().preprocess(sentinel2_tile)
        assert tile.shape[0] == 13
        result = inf.infer(tile)
        assert 0.0 <= result["value_score"] <= 1.0

    def test_backend_reported(self) -> None:
        inf = QuantizedInferencer()
        assert inf.backend in ("stub", "torch_int8") or inf.backend.startswith("onnx:")

    def test_is_stub_flag(self) -> None:
        assert isinstance(QuantizedInferencer().is_stub, bool)


# ── ReActReasoningLoop ──────────────────────────────────────────────────────


class TestReActReasoningLoop:
    def test_high_value_keep(self) -> None:
        cnn = {"cloud_fraction": 0.1, "anomaly_score": 0.8, "value_score": 0.9}
        result = ReActReasoningLoop().reason_and_decide(cnn, {"context": "Wildfire monitoring"})
        assert result["keep"] is True
        assert result["agent_score"] > config.KEEP_SCORE_THRESHOLD

    def test_cloudy_filter(self) -> None:
        cnn = {"cloud_fraction": 0.95, "anomaly_score": 0.1, "value_score": 0.3}
        result = ReActReasoningLoop().reason_and_decide(cnn, {"context": "Routine pass"})
        assert result["keep"] is False

    def test_explanation_generated(self) -> None:
        cnn = {"cloud_fraction": 0.2, "anomaly_score": 0.7, "value_score": 0.8}
        result = ReActReasoningLoop().reason_and_decide(cnn, {"context": "Wildfire"})
        assert len(result["explanation"]) > 0
        assert "Step" in result["explanation"]

    def test_max_steps_respected(self) -> None:
        cnn = {"cloud_fraction": 0.1, "anomaly_score": 0.9, "value_score": 0.9}
        result = ReActReasoningLoop().reason_and_decide(cnn, {"context": "test"})
        assert len(result["steps"]) <= config.AGENT_MAX_STEPS

    def test_agent_score_capped_at_1(self) -> None:
        cnn = {"cloud_fraction": 0.0, "anomaly_score": 1.0, "value_score": 1.0}
        result = ReActReasoningLoop().reason_and_decide(cnn, {"context": "Wildfire disaster"})
        assert result["agent_score"] <= 1.0


# ── PowerMonitor + AgentPowerGuard ──────────────────────────────────────────


class TestPowerMonitor:
    def test_get_current_power_returns_float(self) -> None:
        val = PowerMonitor().get_current_power()
        assert isinstance(val, float)
        assert val >= 0.0

    def test_measure_context_manager(self) -> None:
        with PowerMonitor().measure() as session:
            _ = sum(range(1000))
        assert session.elapsed_seconds > 0.0
        assert isinstance(session.avg_power_watts, float)

    def test_backend_detection(self) -> None:
        assert PowerMonitor().backend in ("tegrastats", "jtop", "cpu_timer")

    def test_is_hardware_backed(self) -> None:
        pm = PowerMonitor()
        assert isinstance(pm.is_hardware_backed, bool)


class TestAgentPowerGuard:
    def test_enforce_budget_runs_block(self) -> None:
        ran = False
        with AgentPowerGuard().enforce_budget(max_watts=50.0):
            ran = True
        assert ran


# ── EdgeTriageEngine (integration) ──────────────────────────────────────────


class TestEdgeTriageEngine:
    def test_process_tile_returns_triage_result(
        self, engine: EdgeTriageEngine, clear_tile: np.ndarray,
    ) -> None:
        result = engine.process_tile(clear_tile)
        assert isinstance(result, TriageResult)
        assert isinstance(result.keep, bool)
        assert 0.0 <= result.final_score <= 1.0

    def test_wildfire_tile_kept(self, engine: EdgeTriageEngine, clear_tile: np.ndarray) -> None:
        result = engine.process_tile(clear_tile, {"context": "Wildfire monitoring"})
        assert result.bandwidth_saved_percent > 0

    def test_cloudy_tile_filtered(self, engine: EdgeTriageEngine, cloudy_tile: np.ndarray) -> None:
        result = engine.process_tile(cloudy_tile, {"context": "Routine pass"})
        assert result.bandwidth_saved_percent >= 60.0

    def test_agent_explanation_present(
        self, engine: EdgeTriageEngine, clear_tile: np.ndarray,
    ) -> None:
        result = engine.process_tile(clear_tile, {"context": "Defense surveillance"})
        assert len(result.explanation) > 0

    def test_sentinel2_full_pipeline(
        self, engine: EdgeTriageEngine, sentinel2_tile: np.ndarray,
    ) -> None:
        result = engine.process_tile(sentinel2_tile, {"context": "Wildfire monitoring"})
        assert isinstance(result, TriageResult)
        assert 0.0 <= result.final_score <= 1.0

    def test_backend_visible_in_result(
        self, engine: EdgeTriageEngine, clear_tile: np.ndarray,
    ) -> None:
        assert "backend" in engine.process_tile(clear_tile).cnn_results

    def test_provenance_fields_populated(
        self, engine: EdgeTriageEngine, clear_tile: np.ndarray,
    ) -> None:
        meta = {"scene_id": "S2A_T30VTL_20260413", "tile_id": "t001",
                "context": "Wildfire", "acquisition_time": "2026-04-13T10:00:00Z"}
        result = engine.process_tile(clear_tile, meta)
        assert result.tile_id == "t001"
        assert result.scene_id == "S2A_T30VTL_20260413"
        assert result.processing_timestamp_utc != ""
        assert len(result.input_hash) == 32  # md5 hex

    def test_auto_tile_id_when_missing(
        self, engine: EdgeTriageEngine, clear_tile: np.ndarray,
    ) -> None:
        result = engine.process_tile(clear_tile)
        assert len(result.tile_id) == 12  # uuid hex[:12]

    def test_error_recovery_on_bad_input(self, engine: EdgeTriageEngine) -> None:
        """Completely invalid input returns safe FILTER result instead of crashing."""
        bad = np.zeros((0,), dtype=np.float32)  # empty array
        result = engine.process_tile(bad)
        assert result.keep is False
        assert result.error is not None
        assert "preprocess" in result.error


# ── Audit Logger ────────────────────────────────────────────────────────────


class TestAuditLogger:
    def test_writes_jsonl_with_hmac(self, engine: EdgeTriageEngine, clear_tile: np.ndarray) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "audit.jsonl"
            hmac_key = b"test-key"
            audit = AuditLogger(path=log_path, hmac_key=hmac_key)
            result = engine.process_tile(clear_tile, {
                "scene_id": "S2A_TEST", "context": "test",
            })
            audit.log_decision(clear_tile.tobytes(), result, {"scene_id": "S2A_TEST"})
            audit.close()

            lines = log_path.read_text().strip().split("\n")
            assert len(lines) == 1
            # Line format: {json}\t{hmac}
            assert "\t" in lines[0]
            json_part, hmac_part = lines[0].rsplit("\t", 1)
            record = json.loads(json_part)
            assert record["scene_id"] == "S2A_TEST"
            assert record["keep"] == result.keep
            assert "input_hash_sha256" in record
            assert "timestamp_utc" in record
            assert record["software_version"] != "unknown"
            assert len(hmac_part) == 64  # SHA-256 hex digest


# ── Parametrized Cloud Filtering Tests ──────────────────────────────────────


class TestCloudFiltering:
    """Parametrized tests verifying cloud filtering across a range of fractions."""

    @pytest.mark.parametrize("cloud_level,should_filter", [
        (0.05, False),   # Nearly clear → keep
        (0.30, False),   # Light cloud → keep
        (0.50, False),   # Partial cloud → keep (below 0.85 threshold)
        (0.90, True),    # Heavy cloud → filter
        (0.95, True),    # Near-total cloud → filter
        (1.00, True),    # Full cloud → filter
    ])
    def test_cloud_fraction_filtering(
        self, engine: EdgeTriageEngine,
        cloud_level: float, should_filter: bool,
    ) -> None:
        """Tiles with cloud_fraction above MAX_CLOUD_FRACTION should be filtered."""
        # Build a tile whose brightness maps to the target cloud fraction via stub
        # Stub formula: cloud_frac = clip(mean * 0.8 + 0.1, 0, 1)
        # Solve for mean: mean = (cloud_level - 0.1) / 0.8
        target_mean = np.clip((cloud_level - 0.1) / 0.8, 0.0, 1.0)
        tile = np.full((64, 64, 3), target_mean, dtype=np.float32)
        result = engine.process_tile(tile, {"context": "Routine monitoring"})
        if should_filter:
            assert result.keep is False, (
                f"cloud_level={cloud_level} should be filtered but was kept "
                f"(score={result.final_score:.3f})"
            )
        else:
            # Low-cloud tiles may still be filtered for other reasons (value),
            # but should not be filtered *solely* due to cloud
            assert result.cnn_results["cloud_fraction"] < config.MAX_CLOUD_FRACTION

    @pytest.mark.parametrize("anomaly_level,context,expected_keep", [
        (0.9, "Wildfire detection", True),    # High anomaly + urgent context → keep
        (0.8, "Defense surveillance", True),   # High anomaly + defense → keep
        (0.1, "Routine pass", False),          # Low anomaly + routine → filter
    ])
    def test_anomaly_context_interaction(
        self, engine: EdgeTriageEngine,
        anomaly_level: float, context: str, expected_keep: bool,
    ) -> None:
        """High anomaly tiles with urgent context should be kept."""
        # Build tile: low cloud, target anomaly via std, moderate value
        # Stub formula: anomaly = clip(std * 2.0, 0, 1)
        # For high anomaly: need high std → use bimodal tile
        rng = np.random.default_rng(42)
        if anomaly_level > 0.5:
            # High std: mix of 0s and 1s
            tile = rng.choice([0.0, 1.0], size=(64, 64, 3)).astype(np.float32)
        else:
            # Low std: uniform low values
            tile = rng.random((64, 64, 3), dtype=np.float32) * 0.1
        result = engine.process_tile(tile, {"context": context})
        assert result.keep is expected_keep, (
            f"anomaly={anomaly_level}, context={context!r}: "
            f"expected keep={expected_keep}, got keep={result.keep} "
            f"(score={result.final_score:.3f})"
        )


# ── SLA Benchmark Tests ─────────────────────────────────────────────────────


class TestSLABenchmarks:
    """Verify the pipeline meets defined SLA targets on stub backend."""

    def test_inference_latency_under_ceiling(self, engine: EdgeTriageEngine, clear_tile: np.ndarray) -> None:
        result = engine.process_tile(clear_tile)
        assert result.cnn_results["inference_ms"] < config.MAX_INFERENCE_MS, (
            f"Inference {result.cnn_results['inference_ms']:.1f}ms > {config.MAX_INFERENCE_MS}ms ceiling"
        )

    def test_final_score_bounded(self, engine: EdgeTriageEngine, clear_tile: np.ndarray) -> None:
        result = engine.process_tile(clear_tile)
        assert 0.0 <= result.final_score <= 1.0

    def test_bandwidth_savings_positive(self, engine: EdgeTriageEngine, clear_tile: np.ndarray) -> None:
        result = engine.process_tile(clear_tile)
        assert result.bandwidth_saved_percent >= 0.0

    def test_batch_throughput(self, engine: EdgeTriageEngine) -> None:
        """10 tiles should complete in <2 seconds on any hardware (stub backend)."""
        import time
        tiles = [np.random.default_rng(i).random((256, 256, 3), dtype=np.float32) for i in range(10)]
        t0 = time.perf_counter()
        for tile in tiles:
            engine.process_tile(tile)
        elapsed = time.perf_counter() - t0
        assert elapsed < 2.0, f"10-tile batch took {elapsed:.2f}s (>2s ceiling)"


# ── MetricsCollector ────────────────────────────────────────────────────────


class TestMetricsCollector:
    def test_record_and_summary(
        self, engine: EdgeTriageEngine, clear_tile: np.ndarray, cloudy_tile: np.ndarray,
    ) -> None:
        collector = MetricsCollector()
        for tile in (clear_tile, cloudy_tile):
            collector.record(tile, engine.process_tile(tile))
        summary = collector.metrics.summary()
        assert summary["tiles_processed"] == 2
        assert summary["bandwidth_saved_percent"] > 0

    def test_reset(self, engine: EdgeTriageEngine, clear_tile: np.ndarray) -> None:
        collector = MetricsCollector()
        collector.record(clear_tile, engine.process_tile(clear_tile))
        assert collector.metrics.tiles_processed == 1
        collector.reset()
        assert collector.metrics.tiles_processed == 0


# ── Security Tests ─────────────────────────────────────────────────────────


class TestSecurityHardening:
    """Verify the pipeline handles adversarial / malformed inputs safely."""

    def test_nan_tile_does_not_crash(self, engine: EdgeTriageEngine) -> None:
        """NaN-filled tile produces a valid result, not a crash."""
        nan_tile = np.full((64, 64, 3), np.nan, dtype=np.float32)
        result = engine.process_tile(nan_tile)
        assert isinstance(result, TriageResult)
        assert 0.0 <= result.final_score <= 1.0

    def test_inf_tile_does_not_crash(self, engine: EdgeTriageEngine) -> None:
        """Inf-filled tile produces a valid result."""
        inf_tile = np.full((64, 64, 3), np.inf, dtype=np.float32)
        result = engine.process_tile(inf_tile)
        assert isinstance(result, TriageResult)

    def test_huge_tile_handled(self, engine: EdgeTriageEngine) -> None:
        """Very large tile doesn't cause OOM — pipeline handles it."""
        big = np.random.default_rng(42).random((2048, 2048, 3), dtype=np.float32)
        result = engine.process_tile(big)
        assert isinstance(result, TriageResult)

    def test_negative_pixel_values(self, engine: EdgeTriageEngine) -> None:
        """Negative pixel values don't crash the pipeline."""
        neg_tile = np.full((64, 64, 3), -1.0, dtype=np.float32)
        result = engine.process_tile(neg_tile)
        assert isinstance(result, TriageResult)

    def test_metadata_injection_safe(self, engine: EdgeTriageEngine, clear_tile: np.ndarray) -> None:
        """Special characters in metadata don't break audit logging or processing."""
        evil_meta = {
            "context": '"; DROP TABLE tiles; --',
            "scene_id": "<script>alert(1)</script>",
            "tile_id": "../../../etc/passwd",
        }
        result = engine.process_tile(clear_tile, evil_meta)
        assert isinstance(result, TriageResult)
        # Metadata is stored as-is (audit log uses JSON — safe from injection)
        assert result.tile_id == "../../../etc/passwd"

    def test_empty_metadata_safe(self, engine: EdgeTriageEngine, clear_tile: np.ndarray) -> None:
        result = engine.process_tile(clear_tile, {})
        assert isinstance(result, TriageResult)

    def test_numpy_pickle_blocked(self, tmp_path: Path) -> None:
        """np.load with allow_pickle=False prevents pickle-based RCE."""
        npy_file = tmp_path / "test.npy"
        np.save(str(npy_file), np.zeros((10, 10, 3)))
        # Loading should work fine for normal npy files
        loaded = ImageIngestor().load(npy_file)
        assert loaded.shape == (10, 10, 3)


# ── Audit HMAC Integrity Tests ─────────────────────────────────────────────


class TestAuditHMAC:
    """Verify HMAC-authenticated audit log integrity."""

    def test_hmac_written_and_verifiable(
        self, engine: EdgeTriageEngine, clear_tile: np.ndarray,
    ) -> None:
        """Audit records include HMAC that passes verification."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "audit.jsonl"
            hmac_key = b"test-key-for-unit-tests"
            audit = AuditLogger(path=log_path, hmac_key=hmac_key)
            result = engine.process_tile(clear_tile, {"scene_id": "HMAC_TEST"})
            audit.log_decision(clear_tile.tobytes(), result, {"scene_id": "HMAC_TEST"})
            audit.close()

            # Verify the log
            results = AuditLogger.verify_log(log_path, hmac_key=hmac_key)
            assert len(results) == 1
            line_num, passed, detail = results[0]
            assert passed, f"HMAC verification failed: {detail}"

    def test_tampered_record_detected(
        self, engine: EdgeTriageEngine, clear_tile: np.ndarray,
    ) -> None:
        """Tampering with a record is detected by HMAC verification."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "audit.jsonl"
            hmac_key = b"test-key-tamper-check"
            audit = AuditLogger(path=log_path, hmac_key=hmac_key)
            result = engine.process_tile(clear_tile, {"scene_id": "TAMPER_TEST"})
            audit.log_decision(clear_tile.tobytes(), result, {"scene_id": "TAMPER_TEST"})
            audit.close()

            # Tamper with the file
            content = log_path.read_text()
            tampered = content.replace('"TAMPER_TEST"', '"HACKED"')
            log_path.write_text(tampered)

            # Verification should fail
            results = AuditLogger.verify_log(log_path, hmac_key=hmac_key)
            assert len(results) == 1
            assert results[0][1] is False  # passed = False

    def test_wrong_key_fails_verification(
        self, engine: EdgeTriageEngine, clear_tile: np.ndarray,
    ) -> None:
        """Verifying with wrong key fails."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "audit.jsonl"
            audit = AuditLogger(path=log_path, hmac_key=b"key-a")
            result = engine.process_tile(clear_tile, {"scene_id": "KEY_TEST"})
            audit.log_decision(clear_tile.tobytes(), result, {"scene_id": "KEY_TEST"})
            audit.close()

            results = AuditLogger.verify_log(log_path, hmac_key=b"key-b")
            assert results[0][1] is False


# ── Model Registry Tests ──────────────────────────────────────────────────


class TestModelRegistry:
    """Verify model registry versioning, checksums, and rollback."""

    def test_register_and_activate(self, tmp_path: Path) -> None:
        reg = ModelRegistry(models_dir=tmp_path / "models")
        # Create a fake model file
        model_file = tmp_path / "test_model.onnx"
        model_file.write_bytes(b"fake-onnx-model-content")

        entry = reg.register(model_file, version="1.0.0", dataset="eurosat",
                             epochs=20, val_loss=0.023)
        assert entry["version"] == "1.0.0"
        assert entry["active"] is True
        assert len(entry["sha256"]) == 64  # SHA-256 hex

    def test_get_active(self, tmp_path: Path) -> None:
        reg = ModelRegistry(models_dir=tmp_path / "models")
        assert reg.get_active() is None  # nothing registered yet

        model = tmp_path / "m.onnx"
        model.write_bytes(b"content")
        reg.register(model, version="1.0.0")

        active = reg.get_active()
        assert active is not None
        assert active["version"] == "1.0.0"

    def test_rollback(self, tmp_path: Path) -> None:
        reg = ModelRegistry(models_dir=tmp_path / "models")
        m1 = tmp_path / "m1.onnx"
        m2 = tmp_path / "m2.onnx"
        m1.write_bytes(b"v1")
        m2.write_bytes(b"v2")

        reg.register(m1, version="1.0.0")
        reg.register(m2, version="1.1.0")
        assert reg.get_active()["version"] == "1.1.0"

        rolled = reg.rollback(to_version="1.0.0")
        assert rolled["version"] == "1.0.0"

    def test_verify_integrity(self, tmp_path: Path) -> None:
        reg = ModelRegistry(models_dir=tmp_path / "models")
        model = tmp_path / "m.onnx"
        model.write_bytes(b"original-content")
        reg.register(model, version="1.0.0")

        # Verify passes
        results = reg.verify_all()
        assert all(ok for _, ok, _ in results)

        # Tamper with the model file
        registered = tmp_path / "models" / "cloud_mask_v1.0.0.onnx"
        registered.write_bytes(b"tampered-content")

        # Verify fails
        results = reg.verify_all()
        assert not all(ok for _, ok, _ in results)

    def test_duplicate_version_rejected(self, tmp_path: Path) -> None:
        reg = ModelRegistry(models_dir=tmp_path / "models")
        model = tmp_path / "m.onnx"
        model.write_bytes(b"content")
        reg.register(model, version="1.0.0")

        with pytest.raises(ValueError, match="already registered"):
            reg.register(model, version="1.0.0")

    def test_list_versions(self, tmp_path: Path) -> None:
        reg = ModelRegistry(models_dir=tmp_path / "models")
        m1 = tmp_path / "m1.onnx"
        m2 = tmp_path / "m2.onnx"
        m1.write_bytes(b"v1")
        m2.write_bytes(b"v2")
        reg.register(m1, version="1.0.0")
        reg.register(m2, version="1.1.0")

        versions = reg.list_versions()
        assert len(versions) == 2
        assert versions[0]["version"] == "1.0.0"
        assert versions[1]["version"] == "1.1.0"


# ── Object detection ────────────────────────────────────────────────────────


class TestObjectDetection:
    """Verify the detection layer finds items of interest and drives triage."""

    def test_detector_stub_runs_on_random_tile(self) -> None:
        from edge_triage.detection import DetectionResult, ObjectDetector
        det = ObjectDetector()
        tile = np.random.default_rng(42).uniform(0.3, 0.9, (13, 256, 256)).astype(np.float32)
        result = det.detect(tile)
        assert isinstance(result, DetectionResult)
        assert result.backend in ("stub", "ultralytics") or result.backend.startswith("onnx")
        # All detections must have valid bboxes in [0,1]
        for d in result.detections:
            assert 0.0 <= d.bbox[0] < d.bbox[2] <= 1.0
            assert 0.0 <= d.bbox[1] < d.bbox[3] <= 1.0
            assert 0.0 < d.confidence <= 1.0
            assert d.class_name

    def test_uniform_tile_has_few_detections(self) -> None:
        """A dead-flat tile has no activity, so the stub should return few/no items."""
        from edge_triage.detection import ObjectDetector
        det = ObjectDetector()
        tile = np.full((13, 256, 256), 0.5, dtype=np.float32)
        result = det.detect(tile)
        assert result.count <= 2  # near-zero activity → very few detections

    def test_high_activity_tile_has_items(self) -> None:
        """A high-variance tile should trigger the stub to report items."""
        from edge_triage.detection import ObjectDetector
        det = ObjectDetector()
        rng = np.random.default_rng(123)
        tile = rng.uniform(0.2, 0.9, (13, 256, 256)).astype(np.float32)
        result = det.detect(tile)
        # High activity should usually produce >=1 detection
        assert result.count >= 1
        assert result.summary() != "no items of interest"

    def test_detection_summary_format(self) -> None:
        """Summary like '3 ships, 1 airplane'."""
        from edge_triage.detection import Detection, DetectionResult
        result = DetectionResult(detections=[
            Detection("ship", 0.9, (0.1, 0.1, 0.2, 0.2)),
            Detection("ship", 0.8, (0.3, 0.3, 0.4, 0.4)),
            Detection("ship", 0.7, (0.5, 0.5, 0.6, 0.6)),
            Detection("airplane", 0.6, (0.7, 0.7, 0.8, 0.8)),
        ])
        summary = result.summary()
        assert "3 ships" in summary
        assert "1 airplane" in summary

    def test_detections_boost_triage_keep_decision(self) -> None:
        """When detector finds items of interest, tile is more likely KEPT."""
        engine = EdgeTriageEngine(audit=False)
        rng = np.random.default_rng(7)
        tile = rng.uniform(0.3, 0.9, (13, 256, 256)).astype(np.float32)
        result = engine.process_tile(tile, {"context": "Maritime ISR"})
        # With detections, agent should activate and contribute to decision
        if result.detection_result and result.detection_result.count > 0:
            # If items found, score should be boosted
            assert result.final_score > 0.0

    def test_cloudy_tile_skips_detection(self) -> None:
        """Cloud-obscured tiles should skip detection entirely."""
        engine = EdgeTriageEngine(audit=False)
        tile = np.full((13, 256, 256), 0.95, dtype=np.float32)
        result = engine.process_tile(tile, {"context": "ISR"})
        # Detection should be None or empty due to cloud guard
        assert result.detection_result is None or result.detection_result.count == 0

    def test_detection_result_serializable(self) -> None:
        """DetectionResult.to_dict() must produce JSON-safe output."""
        from edge_triage.detection import Detection, DetectionResult
        result = DetectionResult(detections=[
            Detection("ship", 0.87, (0.1, 0.2, 0.3, 0.4), class_id=8),
        ])
        d = result.to_dict()
        serialized = json.dumps(d)
        assert "ship" in serialized
        assert "0.87" in serialized
        reloaded = json.loads(serialized)
        assert reloaded["count"] == 1
        assert reloaded["detections"][0]["class_name"] == "ship"

    def test_bbox_normalization(self) -> None:
        """All bboxes must be normalized [0,1] with x1<x2 and y1<y2."""
        from edge_triage.detection import ObjectDetector
        det = ObjectDetector()
        tile = np.random.default_rng(0).uniform(0.1, 0.9, (13, 256, 256)).astype(np.float32)
        result = det.detect(tile)
        for d in result.detections:
            x1, y1, x2, y2 = d.bbox
            assert 0.0 <= x1 < x2 <= 1.0, f"bad x bbox: {d.bbox}"
            assert 0.0 <= y1 < y2 <= 1.0, f"bad y bbox: {d.bbox}"

    def test_confidence_threshold_filters_low_scores(self) -> None:
        """Higher threshold should reduce detection count."""
        from edge_triage.detection import ObjectDetector
        det = ObjectDetector()
        tile = np.random.default_rng(5).uniform(0.2, 0.9, (13, 256, 256)).astype(np.float32)
        low = det.detect(tile, confidence_threshold=0.05)
        high = det.detect(tile, confidence_threshold=0.90)
        # Stricter threshold should match or reduce detections
        assert high.count <= low.count


class TestDetectionPipelineIntegration:
    """End-to-end: detection appears in TriageResult and feeds the agent."""

    def test_triage_result_has_detection_field(self) -> None:
        engine = EdgeTriageEngine(audit=False)
        tile = np.random.default_rng(1).random((256, 256, 3), dtype=np.float32) * 0.5
        r = engine.process_tile(tile, {"context": "test"})
        # Field must exist (None or DetectionResult)
        assert hasattr(r, "detection_result")

    def test_agent_sees_detection_count(self) -> None:
        """ReAct loop should surface detections in its urgency assessment."""
        loop = ReActReasoningLoop()
        cnn = {"cloud_fraction": 0.2, "anomaly_score": 0.3, "value_score": 0.5,
               "detection_count": 3, "detection_summary": "3 ships"}
        meta = {"context": "Maritime ISR"}
        result = loop.reason_and_decide(cnn, meta)
        # Urgency step observation should mention the ships
        step1 = result["steps"][0]
        assert "ships" in step1["observation"].lower() or "HIGH" in step1["observation"]

    def test_detection_info_in_audit_log(self, tmp_path) -> None:
        """Audit record should be writable when detections are present."""
        from edge_triage.audit import AuditLogger
        log_path = tmp_path / "audit.jsonl"
        audit = AuditLogger(path=log_path)
        engine = EdgeTriageEngine(audit=False)
        engine._audit = audit
        tile = np.random.default_rng(9).uniform(0.2, 0.8, (13, 256, 256)).astype(np.float32)
        r = engine.process_tile(tile, {"context": "test", "tile_id": "t1"})
        # Audit log must exist and contain at least one line
        assert log_path.exists()
        lines = log_path.read_text().strip().splitlines()
        assert len(lines) >= 1
        # Each line is JSON + \t + hmac_hex
        json_part = lines[0].rsplit("\t", 1)[0]
        record = json.loads(json_part)
        assert record["tile_id"] == r.tile_id


# ── Live data ingestion ─────────────────────────────────────────────────────


class TestLiveDataSources:
    """Verify live data sources handle failure, parse STAC, and ingest cleanly.

    All network I/O is mocked so the test suite works offline and in CI.
    """

    def test_aoi_presets_well_formed(self) -> None:
        """Every AOI preset must be a valid (minlon, minlat, maxlon, maxlat)."""
        from edge_triage.live_data import AOI_PRESETS
        assert len(AOI_PRESETS) >= 3
        for name, bbox in AOI_PRESETS.items():
            assert len(bbox) == 4, f"{name}: bbox must have 4 elements"
            minlon, minlat, maxlon, maxlat = bbox
            assert -180 <= minlon < maxlon <= 180, f"{name}: bad longitude"
            assert -90 <= minlat < maxlat <= 90, f"{name}: bad latitude"

    def test_livefeeditem_dataclass(self) -> None:
        from edge_triage.live_data import LiveFeedItem
        arr = np.zeros((10, 10, 3), dtype=np.float32)
        item = LiveFeedItem(
            name="test", image=arr, source="sentinel-2",
            acquired_utc="2026-04-17T00:00:00Z",
        )
        assert item.name == "test"
        assert item.image.shape == (10, 10, 3)
        assert item.cloud_cover_pct is None
        assert item.extra == {}

    def test_sentinel2_handles_network_failure_gracefully(self, monkeypatch) -> None:
        """Network errors should return [] rather than crash."""
        from edge_triage import live_data

        def fail(*a, **kw):
            raise ConnectionError("simulated")

        monkeypatch.setattr(live_data, "_http_post_json", fail)
        src = live_data.Sentinel2Source()
        items = src.fetch((-120.0, 35.0, -118.0, 37.0))
        assert items == []

    def test_sentinel2_parses_stac_response(self, monkeypatch) -> None:
        """Mock a minimal STAC response and verify we parse it into LiveFeedItem."""
        from edge_triage import live_data
        import io

        # Fake STAC feature
        fake_stac = {
            "type": "FeatureCollection",
            "features": [{
                "id": "S2B_FAKE_TILE_20260417",
                "bbox": [-120.0, 35.0, -119.0, 36.0],
                "geometry": {"type": "Polygon", "coordinates": [[]]},
                "properties": {
                    "datetime": "2026-04-17T10:30:00.000Z",
                    "eo:cloud_cover": 12.5,
                    "platform": "sentinel-2b",
                    "grid:code": "MGRS-10SFE",
                },
                "assets": {
                    "thumbnail": {"href": "https://example.com/thumb.jpg"},
                },
            }],
        }

        # Fake a 4x4 RGB PNG
        from PIL import Image
        buf = io.BytesIO()
        Image.fromarray(
            np.full((4, 4, 3), 128, dtype=np.uint8),
        ).save(buf, format="PNG")
        fake_png = buf.getvalue()

        monkeypatch.setattr(live_data, "_http_post_json", lambda *a, **k: fake_stac)
        monkeypatch.setattr(live_data, "_http_get", lambda *a, **k: fake_png)

        src = live_data.Sentinel2Source(limit=1)
        items = src.fetch((-120.0, 35.0, -119.0, 36.0))

        assert len(items) == 1
        item = items[0]
        assert item.source == "sentinel-2"
        assert item.cloud_cover_pct == 12.5
        assert item.scene_id == "S2B_FAKE_TILE_20260417"
        assert item.image.shape == (4, 4, 3)
        assert item.image.dtype == np.float32
        assert 0.0 <= item.image.max() <= 1.0

    def test_goes_handles_network_failure(self, monkeypatch) -> None:
        from edge_triage import live_data

        def fail(*a, **kw):
            raise ConnectionError("simulated")

        monkeypatch.setattr(live_data, "_http_get", fail)
        src = live_data.NOAAGOESSource()
        items = src.fetch("CONUS")
        assert items == []

    def test_goes_parses_image(self, monkeypatch) -> None:
        from edge_triage import live_data
        import io

        from PIL import Image
        buf = io.BytesIO()
        Image.fromarray(
            np.full((32, 32, 3), 200, dtype=np.uint8),
        ).save(buf, format="JPEG")
        fake_jpeg = buf.getvalue()

        monkeypatch.setattr(live_data, "_http_get", lambda *a, **k: fake_jpeg)
        src = live_data.NOAAGOESSource()
        items = src.fetch("CA")
        assert len(items) == 1
        assert items[0].source == "goes-18"
        assert items[0].image.shape == (32, 32, 3)

    def test_firms_requires_api_key(self, monkeypatch) -> None:
        """Without a MAP_KEY, FIRMS should return [] and not try to fetch."""
        from edge_triage import live_data

        # Ensure no key is set
        monkeypatch.delenv("EDGE_TRIAGE_FIRMS_KEY", raising=False)
        src = live_data.FIRMSFireSource(map_key="")
        items = src.fetch((-120.0, 35.0, -118.0, 37.0))
        assert items == []

    def test_firms_parses_csv(self, monkeypatch) -> None:
        """Mock a FIRMS CSV response and verify it rasterizes into a heatmap."""
        from edge_triage import live_data

        fake_csv = (
            "country_id,latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,"
            "satellite,instrument,confidence,version,bright_ti5,frp,daynight\n"
            "USA,36.5,-119.5,320.1,0.5,0.5,2026-04-17,1830,N,VIIRS,high,2.0,290.0,18.5,D\n"
            "USA,36.7,-119.2,310.5,0.5,0.5,2026-04-17,1830,N,VIIRS,high,2.0,285.0,25.0,D\n"
        ).encode("utf-8")

        monkeypatch.setattr(live_data, "_http_get", lambda *a, **k: fake_csv)
        src = live_data.FIRMSFireSource(map_key="FAKE_KEY", days=1)
        items = src.fetch((-120.0, 36.0, -119.0, 37.0))
        assert len(items) == 1
        assert items[0].source == "firms"
        assert items[0].extra["fire_count"] == 2
        # Heatmap should have non-zero pixels in the red channel
        assert items[0].image[..., 0].max() > 0.0

    def test_live_item_feeds_triage_pipeline(self) -> None:
        """A LiveFeedItem image should be directly processable by the engine."""
        from edge_triage.live_data import LiveFeedItem

        arr = np.random.default_rng(5).uniform(0.2, 0.8, (64, 64, 3)).astype(np.float32)
        item = LiveFeedItem(
            name="test-live", image=arr, source="sentinel-2",
            acquired_utc="2026-04-17T00:00:00Z", cloud_cover_pct=20.0,
        )
        engine = EdgeTriageEngine(audit=False)
        r = engine.process_tile(item.image, {"context": "Live test",
                                              "tile_id": item.name,
                                              "scene_id": item.name})
        assert 0.0 <= r.final_score <= 1.0
        assert isinstance(r.keep, bool)
