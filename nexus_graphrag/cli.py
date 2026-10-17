"""Command Line Interface (CLI) for Nexus-GraphRAG."""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

from nexus_graphrag.core.models import RetrievalMode
from nexus_graphrag.engine import NexusGraphRAG


def build_parser() -> argparse.ArgumentParser:
    """Constructs command line argument parser."""
    parser = argparse.ArgumentParser(
        prog="nexus-graphrag",
        description="Nexus-GraphRAG: Production Local Knowledge Graph & Hybrid RAG Engine in Python.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Subcommand: index
    index_parser = subparsers.add_parser("index", help="Extract and index a knowledge graph from text or file")
    index_parser.add_argument("--file", "-f", type=str, help="Path to input text document")
    index_parser.add_argument("--text", "-t", type=str, help="Direct text string to index")
    index_parser.add_argument("--output", "-o", type=str, required=True, help="Path to save knowledge graph JSON")
    index_parser.add_argument("--chunk-size", type=int, default=600, help="Chunk character length")
    index_parser.add_argument("--chunk-overlap", type=int, default=100, help="Overlap character length")

    # Subcommand: query
    query_parser = subparsers.add_parser("query", help="Execute Local, Global, or Hybrid knowledge retrieval")
    query_parser.add_argument("--graph", "-g", type=str, required=True, help="Path to indexed knowledge graph JSON")
    query_parser.add_argument("--query", "-q", type=str, required=True, help="Inquiry or research prompt")
    query_parser.add_argument(
        "--mode",
        "-m",
        choices=["local", "global", "hybrid"],
        default="hybrid",
        help="Retrieval modality",
    )

    # Subcommand: visualize
    viz_parser = subparsers.add_parser("visualize", help="Export standalone interactive HTML graph explorer")
    viz_parser.add_argument("--graph", "-g", type=str, required=True, help="Path to indexed knowledge graph JSON")
    viz_parser.add_argument("--output", "-o", type=str, default="knowledge_graph.html", help="Output HTML file path")
    viz_parser.add_argument("--title", type=str, default="Nexus-GraphRAG Knowledge Explorer", help="Report title")

    # Subcommand: stats
    stats_parser = subparsers.add_parser("stats", help="Display graph topology and centrality statistics")
    stats_parser.add_argument("--graph", "-g", type=str, required=True, help="Path to indexed knowledge graph JSON")

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Main CLI execution entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 1

    if args.command == "index":
        text_to_index = ""
        if args.file:
            with open(args.file, "r", encoding="utf-8") as f:
                text_to_index = f.read()
        elif args.text:
            text_to_index = args.text
        else:
            print("[!] Error: Either --file or --text must be specified.", file=sys.stderr)
            return 1

        rag = NexusGraphRAG(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
        summary = rag.index_text(text_to_index)
        rag.save(args.output)
        print(f"[+] Successfully indexed knowledge graph to: {args.output}")
        print(f"    - Entities: {summary['total_entities']}")
        print(f"    - Relationships: {summary['total_relationships']}")
        print(f"    - Communities: {summary['communities_count']}")
        return 0

    elif args.command == "query":
        rag = NexusGraphRAG.load(args.graph)
        result = rag.query(args.query, mode=args.mode)
        print("\n" + "=" * 70)
        print(result.context)
        print("=" * 70)
        print(f"Confidence: {result.confidence_score} | Latency: {result.metrics.get('latency_ms', 0)}ms\n")
        return 0

    elif args.command == "visualize":
        rag = NexusGraphRAG.load(args.graph)
        out_path = rag.export_html(args.output, title=args.title)
        print(f"[+] Interactive HTML graph visualization generated at: {out_path}")
        return 0

    elif args.command == "stats":
        rag = NexusGraphRAG.load(args.graph)
        metrics = rag.get_metrics()
        print("\n" + "=" * 50)
        print("   NEXUS-GRAPHRAG TOPOLOGY METRICS")
        print("=" * 50)
        print(f"Total Nodes (Entities)       : {metrics['total_nodes']}")
        print(f"Total Edges (Relationships)  : {metrics['total_edges']}")
        print(f"Communities Detected         : {len(rag.communities)}")
        print(f"Graph Density                : {metrics['density']}")
        print(f"Average Node Degree          : {metrics['avg_degree']}")
        print(f"Connected Components         : {metrics['connected_components_count']}")
        print("\nTop 5 Entities by PageRank:")
        for name, score in metrics["top_pagerank"][:5]:
            print(f"  • {name:<25} : {score:.4f}")
        print("=" * 50 + "\n")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
