"""Self-contained interactive HTML knowledge graph visualizer with zero external CDN dependencies."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from nexus_graphrag.core.models import Community
from nexus_graphrag.graph.index import GraphIndex
from nexus_graphrag.graph.metrics import compute_graph_metrics

# Color mapping by EntityType
TYPE_COLORS: Dict[str, str] = {
    "TECHNOLOGY": "#38bdf8",    # Sky Cyan
    "ORGANIZATION": "#10b981",  # Emerald
    "PERSON": "#f59e0b",        # Amber
    "CONCEPT": "#a855f7",       # Purple
    "LOCATION": "#ec4899",      # Rose Pink
    "EVENT": "#f97316",         # Orange
    "OTHER": "#94a3b8",         # Slate
}


def generate_html_report(
    index: GraphIndex,
    communities: Optional[List[Community]] = None,
    title: str = "Nexus-GraphRAG Knowledge Explorer",
) -> str:
    """Generates an ultra-responsive, standalone dark-mode HTML dashboard.

    Args:
        index: GraphIndex to visualize.
        communities: List of detected communities.
        title: Title string for report.

    Returns:
        Self-contained HTML file contents as string.
    """
    metrics = compute_graph_metrics(index)
    pr_scores = index.compute_pagerank()
    comms = communities or []

    # Prepare node data for visualizer
    nodes_data: List[Dict[str, Any]] = []
    slug_to_idx: Dict[str, int] = {}

    for idx, (slug, ent) in enumerate(index.entities.items()):
        slug_to_idx[slug] = idx
        pr_val = pr_scores.get(slug, 0.0)
        radius = max(8.0, min(26.0, 10.0 + pr_val * 120.0))
        color = TYPE_COLORS.get(ent.entity_type.value, "#94a3b8")

        nodes_data.append({
            "id": slug,
            "name": ent.name,
            "type": ent.entity_type.value,
            "description": ent.description or "No description provided.",
            "pagerank": round(pr_val, 4),
            "weight": round(ent.weight, 2),
            "color": color,
            "radius": round(radius, 1),
            "degree": index.graph.degree(slug) if slug in index.graph else 0,
        })

    # Prepare link data
    links_data: List[Dict[str, Any]] = []
    for rel in index.relationships.values():
        src_slug = index._name_to_slug.get(rel.source.lower(), rel.source)
        tgt_slug = index._name_to_slug.get(rel.target.lower(), rel.target)

        if src_slug in slug_to_idx and tgt_slug in slug_to_idx:
            links_data.append({
                "source": src_slug,
                "target": tgt_slug,
                "type": rel.relation_type,
                "description": rel.description,
                "weight": round(rel.weight, 2),
            })

    # Communities data
    comms_data = [c.to_dict() for c in comms]

    nodes_json = json.dumps(nodes_data)
    links_json = json.dumps(links_data)
    metrics_json = json.dumps(metrics)
    comms_json = json.dumps(comms_data)

    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{title}</title>
  <style>
    :root {{
      --bg-dark: #090d16;
      --card-bg: #111827;
      --card-border: #1e293b;
      --cyan: #38bdf8;
      --emerald: #10b981;
      --amber: #f59e0b;
      --purple: #a855f7;
      --text-main: #f8fafc;
      --text-muted: #94a3b8;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }}
    body {{ background-color: var(--bg-dark); color: var(--text-main); min-height: 100vh; padding: 24px; }}
    .container {{ max-width: 1400px; margin: 0 auto; }}

    /* Header */
    header {{ display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--card-border); padding-bottom: 18px; margin-bottom: 24px; }}
    .badge {{ background: rgba(56, 189, 248, 0.15); color: var(--cyan); border: 1px solid rgba(56, 189, 248, 0.4); padding: 4px 12px; border-radius: 9999px; font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; }}
    h1 {{ font-size: 26px; font-weight: 800; color: #fff; margin-top: 6px; }}
    .subtitle {{ color: var(--text-muted); font-size: 14px; margin-top: 2px; }}

    /* Stat Cards */
    .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-bottom: 24px; }}
    .stat-card {{ background: var(--card-bg); border: 1px solid var(--card-border); border-radius: 12px; padding: 18px; }}
    .stat-val {{ font-size: 28px; font-weight: 800; color: var(--cyan); }}
    .stat-lbl {{ font-size: 13px; color: var(--text-muted); margin-top: 4px; font-weight: 500; }}

    /* Explorer Layout */
    .explorer-grid {{ display: grid; grid-template-columns: 1fr 340px; gap: 20px; margin-bottom: 24px; }}
    @media (max-width: 980px) {{ .explorer-grid {{ grid-template-columns: 1fr; }} }}

    .canvas-card {{ background: var(--card-bg); border: 1px solid var(--card-border); border-radius: 14px; overflow: hidden; position: relative; }}
    .canvas-toolbar {{ display: flex; justify-content: space-between; align-items: center; padding: 12px 18px; border-bottom: 1px solid var(--card-border); background: rgba(17, 24, 39, 0.8); }}
    .search-input {{ background: #1e293b; border: 1px solid #334155; color: #fff; padding: 6px 14px; border-radius: 8px; font-size: 13px; width: 220px; outline: none; }}
    .search-input:focus {{ border-color: var(--cyan); }}
    #graphCanvas {{ display: block; width: 100%; height: 560px; background: #0c121e; cursor: grab; }}
    #graphCanvas:active {{ cursor: grabbing; }}

    /* Drawer */
    .inspector-card {{ background: var(--card-bg); border: 1px solid var(--card-border); border-radius: 14px; padding: 20px; display: flex; flex-direction: column; }}
    .inspector-title {{ font-size: 17px; font-weight: 700; margin-bottom: 12px; color: #fff; border-bottom: 1px solid var(--card-border); padding-bottom: 8px; }}
    .detail-item {{ margin-bottom: 10px; font-size: 13px; }}
    .detail-label {{ color: var(--text-muted); font-size: 11px; text-transform: uppercase; font-weight: 600; letter-spacing: 0.05em; }}
    .detail-value {{ color: #e2e8f0; font-weight: 600; margin-top: 2px; }}

    /* Tables */
    .tables-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
    @media (max-width: 900px) {{ .tables-grid {{ grid-template-columns: 1fr; }} }}
    .table-card {{ background: var(--card-bg); border: 1px solid var(--card-border); border-radius: 12px; padding: 18px; overflow-x: auto; }}
    .table-title {{ font-size: 16px; font-weight: 700; margin-bottom: 14px; color: #fff; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; text-align: left; }}
    th {{ color: var(--text-muted); font-size: 11px; text-transform: uppercase; border-bottom: 1px solid var(--card-border); padding: 8px 10px; }}
    td {{ padding: 10px; border-bottom: 1px solid rgba(255, 255, 255, 0.04); color: #cbd5e1; }}
    tr:hover td {{ background: rgba(56, 189, 248, 0.04); }}

    /* Footer */
    footer {{ text-align: center; margin-top: 36px; padding-top: 20px; border-top: 1px solid var(--card-border); color: var(--text-muted); font-size: 12px; }}
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div>
        <span class="badge">Nexus-GraphRAG Local Engine</span>
        <h1>{title}</h1>
        <div class="subtitle">Multi-Hop Knowledge Graph Topology, Community Modularity & Entity Intelligence</div>
      </div>
      <div>
        <span style="color: var(--emerald); font-weight: 700; font-size: 13px;">● 100% Offline Standalone</span>
      </div>
    </header>

    <!-- Metrics Row -->
    <div class="stats-grid">
      <div class="stat-card">
        <div class="stat-val">{metrics['total_nodes']}</div>
        <div class="stat-lbl">Extracted Entities</div>
      </div>
      <div class="stat-card">
        <div class="stat-val" style="color: var(--emerald);">{metrics['total_edges']}</div>
        <div class="stat-lbl">Relational Edges</div>
      </div>
      <div class="stat-card">
        <div class="stat-val" style="color: var(--purple);">{len(comms)}</div>
        <div class="stat-lbl">Detected Communities</div>
      </div>
      <div class="stat-card">
        <div class="stat-val" style="color: var(--amber);">{metrics['density']}</div>
        <div class="stat-lbl">Graph Density</div>
      </div>
      <div class="stat-card">
        <div class="stat-val">{metrics['avg_degree']}</div>
        <div class="stat-lbl">Average Degree</div>
      </div>
    </div>

    <!-- Main Explorer Canvas + Inspector -->
    <div class="explorer-grid">
      <div class="canvas-card">
        <div class="canvas-toolbar">
          <span style="font-size: 13px; font-weight: 600; color: #cbd5e1;">Interactive Force Graph</span>
          <input type="text" id="searchBox" class="search-input" placeholder="Search entity name...">
        </div>
        <canvas id="graphCanvas"></canvas>
      </div>

      <div class="inspector-card">
        <div class="inspector-title" id="inspName">Entity Inspector</div>
        <div class="detail-item">
          <div class="detail-label">Entity Type</div>
          <div class="detail-value" id="inspType">-</div>
        </div>
        <div class="detail-item">
          <div class="detail-label">PageRank Centrality</div>
          <div class="detail-value" id="inspPR" style="color: var(--cyan);">-</div>
        </div>
        <div class="detail-item">
          <div class="detail-label">Degree (Connections)</div>
          <div class="detail-value" id="inspDeg">-</div>
        </div>
        <div class="detail-item">
          <div class="detail-label">Description</div>
          <div class="detail-value" id="inspDesc" style="font-weight: 400; font-size: 12px; line-height: 1.4; color: #94a3b8;">
            Click on any node in the canvas to inspect its relational attributes and multi-hop connections.
          </div>
        </div>
        <div class="detail-item" style="margin-top: 10px;">
          <div class="detail-label">Connected Edges</div>
          <div id="inspEdges" style="font-size: 11px; color: #cbd5e1; max-height: 160px; overflow-y: auto; margin-top: 4px;">
            -
          </div>
        </div>
      </div>
    </div>

    <!-- Data Tables Row -->
    <div class="tables-grid">
      <div class="table-card">
        <div class="table-title">Top Central Entities (PageRank Leaderboard)</div>
        <table>
          <thead>
            <tr>
              <th>Rank</th>
              <th>Entity</th>
              <th>Type</th>
              <th>Score</th>
            </tr>
          </thead>
          <tbody id="prTableBody"></tbody>
        </table>
      </div>

      <div class="table-card">
        <div class="table-title">Detected Knowledge Communities (Louvain Clusters)</div>
        <table>
          <thead>
            <tr>
              <th>Cluster</th>
              <th>Title</th>
              <th>Size</th>
              <th>Key Members</th>
            </tr>
          </thead>
          <tbody id="commTableBody"></tbody>
        </table>
      </div>
    </div>

    <footer>
      Nexus-GraphRAG Engine • Developed by Nitesh Beniwal • Standalone Report Generated with Zero External CDN Dependencies
    </footer>
  </div>

  <script>
    const NODES = {nodes_json};
    const LINKS = {links_json};
    const COMMS = {comms_json};

    // Populate PageRank Table
    const prTbody = document.getElementById('prTableBody');
    const sortedNodes = [...NODES].sort((a, b) => b.pagerank - a.pagerank).slice(0, 8);
    sortedNodes.forEach((n, idx) => {{
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td style="font-weight: 700; color: var(--cyan);">#${{idx + 1}}</td>
        <td style="font-weight: 600; color: #fff;">${{n.name}}</td>
        <td><span style="font-size: 11px; padding: 2px 6px; border-radius: 4px; background: rgba(255,255,255,0.06); color: ${{n.color}};">${{n.type}}</span></td>
        <td style="font-family: monospace; color: var(--emerald);">${{n.pagerank.toFixed(4)}}</td>
      `;
      prTbody.appendChild(tr);
    }});

    // Populate Communities Table
    const commTbody = document.getElementById('commTableBody');
    COMMS.forEach((c, idx) => {{
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td style="font-weight: 700; color: var(--purple);">C-${{idx + 1}}</td>
        <td style="font-weight: 600; color: #fff;">${{c.title}}</td>
        <td>${{c.member_entities.length}} nodes</td>
        <td style="font-size: 12px; color: var(--text-muted);">${{c.member_entities.slice(0, 3).join(', ')}}</td>
      `;
      commTbody.appendChild(tr);
    }});

    // Canvas Simulation Setup
    const canvas = document.getElementById('graphCanvas');
    const ctx = canvas.getContext('2d');
    let width, height;

    function resizeCanvas() {{
      width = canvas.parentElement.clientWidth;
      height = 560;
      canvas.width = width * window.devicePixelRatio;
      canvas.height = height * window.devicePixelRatio;
      canvas.style.width = width + 'px';
      canvas.style.height = height + 'px';
      ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
    }}
    window.addEventListener('resize', () => {{ resizeCanvas(); render(); }});
    resizeCanvas();

    // Node Positions Initialization
    const nodeMap = {{}};
    NODES.forEach((n, i) => {{
      const angle = (i / Math.max(1, NODES.length)) * 2 * Math.PI;
      const radius = Math.min(width, height) * 0.32;
      n.x = width / 2 + Math.cos(angle) * radius + (Math.random() - 0.5) * 40;
      n.y = height / 2 + Math.sin(angle) * radius + (Math.random() - 0.5) * 40;
      n.vx = 0;
      n.vy = 0;
      nodeMap[n.id] = n;
    }});

    // Simple Force-Directed Relaxation Step
    function tickPhysics() {{
      // Repulsion
      for (let i = 0; i < NODES.length; i++) {{
        for (let j = i + 1; j < NODES.length; j++) {{
          const a = NODES[i], b = NODES[j];
          let dx = b.x - a.x;
          let dy = b.y - a.y;
          let dist = Math.sqrt(dx * dx + dy * dy) || 1;
          if (dist < 220) {{
            const force = (220 - dist) / dist * 0.12;
            a.vx -= dx * force;
            a.vy -= dy * force;
            b.vx += dx * force;
            b.vy += dy * force;
          }}
        }}
      }}

      // Link Spring Attraction
      LINKS.forEach(l => {{
        const src = nodeMap[l.source];
        const tgt = nodeMap[l.target];
        if (src && tgt) {{
          let dx = tgt.x - src.x;
          let dy = tgt.y - src.y;
          let dist = Math.sqrt(dx * dx + dy * dy) || 1;
          let force = (dist - 90) * 0.008;
          src.vx += dx * force;
          src.vy += dy * force;
          tgt.vx -= dx * force;
          tgt.vy -= dy * force;
        }}
      }});

      // Center Gravity & Damping
      NODES.forEach(n => {{
        n.vx += (width / 2 - n.x) * 0.004;
        n.vy += (height / 2 - n.y) * 0.004;
        n.vx *= 0.88;
        n.vy *= 0.88;
        n.x += n.vx;
        n.y += n.vy;
      }});
    }}

    // Run 60 simulation steps for initial stabilization
    for (let s = 0; s < 60; s++) tickPhysics();

    // Canvas Rendering
    let selectedNode = null;
    let hoveredNode = null;
    let searchQuery = '';

    function render() {{
      ctx.clearRect(0, 0, width, height);

      // Draw Edges
      LINKS.forEach(l => {{
        const src = nodeMap[l.source];
        const tgt = nodeMap[l.target];
        if (!src || !tgt) return;

        const isHighlighted = (selectedNode && (src.id === selectedNode.id || tgt.id === selectedNode.id));
        ctx.beginPath();
        ctx.moveTo(src.x, src.y);
        ctx.lineTo(tgt.x, tgt.y);
        ctx.strokeStyle = isHighlighted ? 'rgba(56, 189, 248, 0.9)' : 'rgba(100, 116, 139, 0.28)';
        ctx.lineWidth = isHighlighted ? 2.2 : 1.2;
        ctx.stroke();

        if (isHighlighted) {{
          const midX = (src.x + tgt.x) / 2;
          const midY = (src.y + tgt.y) / 2;
          ctx.fillStyle = '#94a3b8';
          ctx.font = '10px sans-serif';
          ctx.fillText(l.type, midX + 4, midY - 4);
        }}
      }});

      // Draw Nodes
      NODES.forEach(n => {{
        const isMatch = !searchQuery || n.name.toLowerCase().includes(searchQuery.toLowerCase());
        const isSelected = selectedNode && selectedNode.id === n.id;
        const radius = isSelected ? n.radius + 4 : n.radius;

        ctx.beginPath();
        ctx.arc(n.x, n.y, radius, 0, Math.PI * 2);
        ctx.fillStyle = isMatch ? n.color : 'rgba(71, 85, 105, 0.4)';
        ctx.fill();

        if (isSelected) {{
          ctx.lineWidth = 3;
          ctx.strokeStyle = '#ffffff';
          ctx.stroke();
        }}

        // Label
        if (n.pagerank > 0.03 || isSelected || (hoveredNode && hoveredNode.id === n.id)) {{
          ctx.fillStyle = '#ffffff';
          ctx.font = isSelected ? 'bold 12px sans-serif' : '11px sans-serif';
          ctx.fillText(n.name, n.x + radius + 4, n.y + 4);
        }}
      }});
    }}

    render();

    // Interaction Handlers
    function findNodeAt(mx, my) {{
      for (let i = NODES.length - 1; i >= 0; i--) {{
        const n = NODES[i];
        const dist = Math.hypot(n.x - mx, n.y - my);
        if (dist <= n.radius + 6) return n;
      }}
      return null;
    }}

    canvas.addEventListener('mousemove', e => {{
      const rect = canvas.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;
      const hit = findNodeAt(mx, my);
      if (hit !== hoveredNode) {{
        hoveredNode = hit;
        render();
      }}
    }});

    canvas.addEventListener('click', e => {{
      const rect = canvas.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;
      const hit = findNodeAt(mx, my);
      if (hit) {{
        selectedNode = hit;
        updateInspector(hit);
      }} else {{
        selectedNode = null;
      }}
      render();
    }});

    function updateInspector(node) {{
      document.getElementById('inspName').textContent = node.name;
      document.getElementById('inspType').textContent = node.type;
      document.getElementById('inspType').style.color = node.color;
      document.getElementById('inspPR').textContent = node.pagerank.toFixed(4);
      document.getElementById('inspDeg').textContent = `${{node.degree}} incident edges`;
      document.getElementById('inspDesc').textContent = node.description;

      const incidentLinks = LINKS.filter(l => l.source === node.id || l.target === node.id);
      const edgesContainer = document.getElementById('inspEdges');
      if (incidentLinks.length === 0) {{
        edgesContainer.textContent = 'No incident edges recorded.';
      }} else {{
        edgesContainer.innerHTML = incidentLinks.map(l => {{
          const isSrc = l.source === node.id;
          const other = isSrc ? nodeMap[l.target]?.name || l.target : nodeMap[l.source]?.name || l.source;
          return `<div style="padding: 3px 0; border-bottom: 1px solid #1e293b;">
            <span style="color: var(--cyan); font-weight: 600;">${{isSrc ? '→' : '←'}} ${{l.type}}</span> ${{other}}
          </div>`;
        }}).join('');
      }}
    }}

    // Search filter
    document.getElementById('searchBox').addEventListener('input', e => {{
      searchQuery = e.target.value;
      render();
    }});
  </script>
</body>
</html>"""
    return html_template
