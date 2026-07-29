from pathlib import Path


def test_submitted_reference_script_is_preserved() -> None:
    root = Path(__file__).resolve().parents[1]
    path = (
        root
        / "third_party"
        / "sfo_reference"
        / "three_phase_sfo_reference.py"
    )
    text = path.read_text(encoding="utf-8")
    assert "Final Experiment: Correct Three-Phase" in text
    assert "W_GLIDE = 0.35" in text
    assert "TARGET_IMPROVEMENT_WINDOW = 30" in text
    assert "MICRO_STAGNATION_LIMIT = 15" in text
