"""Tests for command-line interface."""

import os
import tempfile
from nexus_graphrag.cli import main


def test_cli_help(capsys):
    ret = main([])
    assert ret == 1


def test_cli_index_query_stats_visualize():
    with tempfile.TemporaryDirectory() as tmp_dir:
        input_txt = os.path.join(tmp_dir, "input.txt")
        graph_json = os.path.join(tmp_dir, "graph.json")
        report_html = os.path.join(tmp_dir, "report.html")

        with open(input_txt, "w", encoding="utf-8") as f:
            f.write("Google DeepMind developed AlphaFold in London. AlphaFold uses PyTorch.")

        # 1. Test Index
        ret_idx = main(["index", "--file", input_txt, "--output", graph_json])
        assert ret_idx == 0
        assert os.path.exists(graph_json)

        # 2. Test Stats
        ret_stats = main(["stats", "--graph", graph_json])
        assert ret_stats == 0

        # 3. Test Query
        ret_query = main(["query", "--graph", graph_json, "--query", "What is AlphaFold?", "--mode", "hybrid"])
        assert ret_query == 0

        # 4. Test Visualize
        ret_viz = main(["visualize", "--graph", graph_json, "--output", report_html])
        assert ret_viz == 0
        assert os.path.exists(report_html)
