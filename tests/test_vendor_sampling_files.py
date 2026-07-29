from pathlib import Path


def test_all_vendor_sampling_files_exist() -> None:
    root = Path(__file__).resolve().parents[1]
    required = [
        "third_party/pythonrobotics/PathPlanning/RRT/rrt.py",
        "third_party/pythonrobotics/PathPlanning/RRTStar/rrt_star.py",
        (
            "third_party/pythonrobotics/PathPlanning/"
            "InformedRRTStar/informed_rrt_star.py"
        ),
        (
            "third_party/pythonrobotics/PathPlanning/"
            "ProbabilisticRoadMap/probabilistic_road_map.py"
        ),
        (
            "third_party/pythonrobotics/PathPlanning/"
            "ThetaStar/theta_star.py"
        ),
    ]
    assert all((root / path).is_file() for path in required)
