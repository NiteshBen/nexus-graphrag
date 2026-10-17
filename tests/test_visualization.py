"""Tests for standalone HTML knowledge graph visualizer."""

import os
import tempfile

from nexus_graphrag.core.models import Community
from nexus_graphrag.graph.index import GraphIndex
from nexus_graphrag.visualizer.html_report import generate_html_report


def test_generate_html_report(populated_index: GraphIndex):
    comms = [
        Community(
            id="comm_0",
            level=0,
            title="AI Modeling Community",
            summary="A test community cluster.",
            member_entities=["AlphaFold", "PyTorch"],
        )
    ]

    html_content = generate_html_report(
        populated_index,
        communities=comms,
        title="Test Explorer",
    )

    assert "<!DOCTYPE html>" in html_content
    assert "Test Explorer" in html_content
    assert "AlphaFold" in html_content
    assert "Google DeepMind" in html_content
    assert "Interactive Force Graph" in html_content

    # Strict Zero-CDN verification: verify no external scripts/stylesheets are loaded
    assert "http://" not in html_content
    assert "https://" not in html_content
    assert "cdn." not in html_content
    assert "unpkg.com" not in html_content
    assert "cdnjs." not in html_content


def test_html_export_to_file(populated_index: GraphIndex):
    with tempfile.TemporaryDirectory() as tmp_dir:
        report_path = os.path.join(tmp_dir, "graph_report.html")
        html_str = generate_html_report(populated_index)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html_str)

        assert os.path.exists(report_path)
        assert os.path.getsize(report_path) > 1000
