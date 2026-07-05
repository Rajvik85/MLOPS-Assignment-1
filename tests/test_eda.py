"""
Tests for the reproducible EDA plot generation script.
"""

from pathlib import Path

from src.eda import generate_eda_plots


def test_generate_eda_plots_creates_required_files(tmp_path):
    raw_data = tmp_path / "processed.cleveland.data"
    raw_data.write_text(
        "\n".join(
            [
                "63.0,1.0,1.0,145.0,233.0,1.0,2.0,150.0,0.0,2.3,3.0,0.0,6.0,0",
                "67.0,1.0,4.0,160.0,286.0,0.0,2.0,108.0,1.0,1.5,2.0,3.0,3.0,2",
                "67.0,1.0,4.0,120.0,229.0,0.0,2.0,129.0,1.0,2.6,2.0,2.0,7.0,1",
                "37.0,1.0,3.0,130.0,250.0,0.0,0.0,187.0,0.0,3.5,3.0,0.0,3.0,0",
                "41.0,0.0,2.0,130.0,204.0,0.0,2.0,172.0,0.0,1.4,1.0,?,3.0,0",
                "56.0,1.0,2.0,120.0,236.0,0.0,0.0,178.0,0.0,0.8,1.0,0.0,?,1",
            ]
        )
    )
    output_dir = tmp_path / "plots"

    generated_files = generate_eda_plots(str(raw_data), str(output_dir))

    assert len(generated_files) == 5
    assert {Path(file_path).name for file_path in generated_files} == {
        "eda_age_thalach_relationship.png",
        "eda_class_balance.png",
        "eda_correlation_heatmap.png",
        "eda_missing_values.png",
        "eda_numerical_histograms.png",
    }
    for file_path in generated_files:
        assert Path(file_path).exists()
        assert Path(file_path).stat().st_size > 0
