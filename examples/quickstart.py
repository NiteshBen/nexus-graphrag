import os
import sys

# Set UTF-8 encoding for Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure package is resolvable from examples/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nexus_graphrag import NexusGraphRAG

SAMPLE_CORPUS = """
Nexus-GraphRAG is an open-source Python framework developed by Nitesh Beniwal in Jaipur, Rajasthan.
It bridges the gap between unstructured document collections and structured graph reasoning.
The architecture combines NetworkX, NumPy, and Scikit-learn algorithms with modern Generative AI pipelines.

Traditional vector databases struggle with holistic questions and multi-hop relationships.
Nexus-GraphRAG solves this by implementing dual-mode extraction: a 100% offline deterministic extractor
and an optional Google Gemini 2.0 Flash LLM extractor.
The engine segments documents into overlapping chunks, discovers entities, and builds typed relational edges.

Once the knowledge graph is constructed, Louvain modularity partitions interconnected nodes into semantic communities.
Nexus-GraphRAG supports three retrieval modes:
1. Local Retrieval: Focuses on entity ego-networks and multi-hop path traversal.
2. Global Retrieval: Synthesizes high-level thematic abstracts across community clusters.
3. Hybrid Retrieval: Uses Reciprocal Rank Fusion (RRF) to blend local paths, community summaries, and BM25 chunk evidence.

Nexus-GraphRAG integrates with EvalGuard AI to evaluate RAG faithfulness, answer relevance, and context recall.
It also pairs with SynthoTab for privacy-preserving synthetic dataset generation using Differential Privacy.
Finally, Nexus-GraphRAG exports standalone, responsive dark-mode HTML reports with interactive force graphs.
"""


def main():
    print("=" * 70)
    print("   NEXUS-GRAPHRAG QUICKSTART DEMONSTRATION")
    print("=" * 70)

    # 1. Initialize Engine
    print("\n[1] Initializing Nexus-GraphRAG Engine...")
    rag = NexusGraphRAG(chunk_size=400, chunk_overlap=80)

    # 2. Ingest & Index Text
    print("[2] Indexing technical corpus...")
    summary = rag.index_text(SAMPLE_CORPUS, doc_id="enterprise_brief")
    print(f"    ✓ Chunks Added        : {summary['chunks_added']}")
    print(f"    ✓ Extracted Entities  : {summary['total_entities']}")
    print(f"    ✓ Extracted Relations : {summary['total_relationships']}")
    print(f"    ✓ Communities Detected: {summary['communities_count']}")

    # 3. Print Topology Metrics
    metrics = rag.get_metrics()
    print(f"    ✓ Graph Density       : {metrics['density']}")
    print(f"    ✓ Average Degree      : {metrics['avg_degree']}")
    print("\n[3] Top 5 Entities by PageRank Centrality:")
    for name, pr in metrics["top_pagerank"][:5]:
        print(f"    • {name:<22} : {pr:.4f}")

    # 4. Execute Local Query
    print("\n" + "-" * 70)
    print("[4] Executing Local Multi-Hop Retrieval: 'How does Nexus-GraphRAG relate to EvalGuard AI?'")
    local_res = rag.query("How does Nexus-GraphRAG relate to EvalGuard AI?", mode="local")
    print(local_res.context)

    # 5. Execute Global Query
    print("\n" + "-" * 70)
    print("[5] Executing Global Thematic Retrieval: 'Summarize overarching architecture and communities'")
    global_res = rag.query("Summarize overarching architecture and communities", mode="global")
    print(global_res.context)

    # 6. Execute Hybrid Query
    print("\n" + "-" * 70)
    print("[6] Executing Hybrid Retrieval (RRF Fusion): 'What retrieval modes are supported by Nexus-GraphRAG?'")
    hybrid_res = rag.query("What retrieval modes are supported by Nexus-GraphRAG?", mode="hybrid")
    print(hybrid_res.context)

    # 7. Export Interactive HTML Report
    html_path = os.path.join(os.path.dirname(__file__), "nexus_graphrag_demo.html")
    rag.export_html(html_path, title="Nexus-GraphRAG Demo Explorer")
    print(f"\n[7] Standalone interactive HTML report saved to: {html_path}")

    # 8. Save Graph to JSON
    json_path = os.path.join(os.path.dirname(__file__), "nexus_graphrag_demo.json")
    rag.save(json_path)
    print(f"[8] Graph state serialized to: {json_path}")
    print("\n" + "=" * 70)
    print("   DEMONSTRATION COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
