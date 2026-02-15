#!/usr/bin/env python
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def _load_trace_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows


def _derive_forensics(trace_rows: List[Dict[str, Any]], meta: Dict[str, Any]) -> Dict[str, Any]:
    if len(trace_rows) == 0:
        return {
            "cause_localization_drift": False,
            "cause_controller_oscillation": False,
            "cause_fwd_no_motion": False,
            "cause_dataset_reachability": False,
            "fwd_no_motion_count": 0,
            "watchdog_count": 0,
            "mean_entropy": 0.0,
            "node_switches": 0,
            "lr_switch_count": 0,
        }
    ents = [float(r.get("belief_entropy", 0.0)) for r in trace_rows]
    mean_ent = sum(ents) / max(1, len(ents))
    fwd_no_motion = sum(int(r.get("fwd_no_motion", 0)) for r in trace_rows)
    watchdog = 0
    lr_switch = 0
    node_switches = 0
    prev_node = None
    prev_action = None
    for r in trace_rows:
        ev = r.get("recovery_event", []) or []
        if any("watchdog" in str(x) for x in ev):
            watchdog += 1
        node = int(r.get("map_node_id", -1))
        if prev_node is not None and node != prev_node:
            node_switches += 1
        prev_node = node
        act = str(r.get("action", ""))
        if prev_action in {"LEFT", "RIGHT"} and act in {"LEFT", "RIGHT"} and act != prev_action:
            lr_switch += 1
        prev_action = act

    oracle = meta.get("oracle", {}) if isinstance(meta.get("oracle"), dict) else {}
    oracle_available = int(oracle.get("oracle_available", 0)) == 1
    oracle_success = int(oracle.get("oracle_success", 0)) == 1
    geo_start_raw = oracle.get("oracle_geodesic_start", float("inf"))
    try:
        geo_start = float(geo_start_raw) if geo_start_raw is not None else float("inf")
    except Exception:
        geo_start = float("inf")
    dataset_issue = oracle_available and (not oracle_success) and (not (geo_start < 1e9))
    localization_drift = (mean_ent >= 0.93 and node_switches >= max(3, len(trace_rows) // 10))
    controller_osc = (lr_switch >= max(5, len(trace_rows) // 12)) or (watchdog >= 3)
    fwd_blocked = fwd_no_motion >= 3
    return {
        "cause_localization_drift": bool(localization_drift),
        "cause_controller_oscillation": bool(controller_osc),
        "cause_fwd_no_motion": bool(fwd_blocked),
        "cause_dataset_reachability": bool(dataset_issue),
        "fwd_no_motion_count": int(fwd_no_motion),
        "watchdog_count": int(watchdog),
        "mean_entropy": float(mean_ent),
        "node_switches": int(node_switches),
        "lr_switch_count": int(lr_switch),
    }


def _build_graph_payload(build_dir: Path) -> Dict[str, Any]:
    graph = _load_json(build_dir / "graph.json", {"nodes": [], "edges": []})
    room_graph = _load_json(build_dir / "room_graph.json", {"place2room": {}, "doorway_debug": {}})
    node_to_room = _load_json(build_dir / "node_to_room.json", {"node_to_room": {}})
    n2r = {int(k): int(v) for k, v in (node_to_room.get("node_to_room", {}) or {}).items()}
    if not n2r:
        n2r = {int(k): int(v) for k, v in (room_graph.get("place2room", {}) or {}).items()}
    nodes_out = []
    for n in graph.get("nodes", []):
        nid = int(n.get("node_id", -1))
        pos = n.get("position", [0.0, 0.0, 0.0])
        nodes_out.append(
            {
                "id": nid,
                "x": float(pos[0]),
                "z": float(pos[2]),
                "room_id": int(n2r.get(nid, -1)),
            }
        )
    edges_out = []
    for e in graph.get("edges", []):
        src = e.get("source", e.get("u"))
        dst = e.get("target", e.get("v"))
        if src is None or dst is None:
            continue
        edges_out.append(
            {
                "source": int(src),
                "target": int(dst),
                "weight": float(e.get("weight", e.get("w", 0.0))),
                "edge_type": str(e.get("edge_type", "temporal")),
            }
        )
    doorway = room_graph.get("doorway_debug", {})
    door_edges = []
    for row in doorway.get("doorway_edges", []) or []:
        if isinstance(row, (list, tuple)) and len(row) >= 2:
            door_edges.append([int(row[0]), int(row[1])])
    return {
        "nodes": nodes_out,
        "edges": edges_out,
        "doorway_edges": door_edges,
        "place2room": {str(k): int(v) for k, v in n2r.items()},
    }


HTML_TEMPLATE = """<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Topo Debug Report</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 0; background: #f5f6f8; }
    .header { padding: 12px 16px; background: #111827; color: #fff; }
    .layout { display: grid; grid-template-columns: 34% 33% 33%; gap: 10px; padding: 10px; }
    .card { background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 10px; }
    .row { margin: 4px 0; font-size: 13px; }
    #frame { width: 100%; max-height: 340px; object-fit: contain; background: #000; }
    canvas { border: 1px solid #ddd; width: 100%; height: 320px; }
    .small { font-size: 12px; color: #333; white-space: pre-wrap; }
    .kpi { font-weight: 700; }
    .plots canvas { height: 120px; margin-top: 6px; }
  </style>
</head>
<body>
  <div class="header"><span id="title">Topo Debug Report</span></div>
  <div class="layout">
    <div class="card">
      <div class="row"><span class="kpi">Run</span>: <span id="runInfo"></span></div>
      <div class="row"><span class="kpi">Termination</span>: <span id="termInfo"></span></div>
      <div class="row"><span class="kpi">Oracle</span>: <span id="oracleInfo"></span></div>
      <div class="row"><span class="kpi">Forensics</span>: <span id="forensicsInfo"></span></div>
      <div class="row"><span class="kpi">Goal</span>: <span id="goalInfo"></span></div>
      <div class="row"><span class="kpi">Renderer</span>: <span id="rendererInfo"></span></div>
      <div class="row"><span class="kpi">RGB Capture</span>: <span id="rgbCaptureInfo"></span></div>
      <input id="stepSlider" type="range" min="0" max="0" value="0" style="width:100%"/>
      <div class="row">Step: <span id="stepText"></span></div>
      <img id="frame" src="" alt="frame"/>
      <div class="small" id="telemetry"></div>
    </div>
    <div class="card">
      <div class="row"><span class="kpi">Topo/Room Map</span></div>
      <canvas id="mapCanvas" width="640" height="480"></canvas>
      <div class="small" id="mapLegend"></div>
    </div>
    <div class="card plots">
      <div class="row"><span class="kpi">Telemetry Curves</span></div>
      <canvas id="locPlot" width="620" height="120"></canvas>
      <canvas id="goalPlot" width="620" height="120"></canvas>
      <canvas id="subgoalPlot" width="620" height="120"></canvas>
      <div class="small" id="beliefTopk"></div>
    </div>
  </div>
<script>
async function loadAll() {
  const [meta, trace, graph] = await Promise.all([
    fetch('meta.json').then(r=>r.json()),
    fetch('trace.json').then(r=>r.json()),
    fetch('graph.json').then(r=>r.json()),
  ]);
  return {meta, trace, graph};
}
function colorForRoom(id){
  const pal=['#4f46e5','#0891b2','#16a34a','#f59e0b','#dc2626','#7c3aed','#0ea5e9','#65a30d','#fb7185','#374151'];
  if(id<0) return '#9ca3af';
  return pal[id % pal.length];
}
function drawSeries(canvasId, vals, minY, maxY, color){
  const c=document.getElementById(canvasId),ctx=c.getContext('2d');
  ctx.clearRect(0,0,c.width,c.height);
  ctx.strokeStyle='#e5e7eb'; ctx.beginPath(); ctx.moveTo(0,c.height-1); ctx.lineTo(c.width,c.height-1); ctx.stroke();
  if(!vals || vals.length===0) return;
  const n=vals.length;
  const y0=isFinite(minY)?minY:Math.min(...vals), y1=isFinite(maxY)?maxY:Math.max(...vals)+1e-6;
  ctx.strokeStyle=color; ctx.beginPath();
  for(let i=0;i<n;i++){
    const x=(i/(Math.max(1,n-1)))*(c.width-1);
    const t=(vals[i]-y0)/(Math.max(1e-6,y1-y0));
    const y=(1-t)*(c.height-6)+3;
    if(i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
  }
  ctx.stroke();
}
function drawMap(graph, trace, step, meta){
  const c=document.getElementById('mapCanvas'),ctx=c.getContext('2d');
  ctx.clearRect(0,0,c.width,c.height);
  const nodes=graph.nodes||[], edges=graph.edges||[], door=graph.doorway_edges||[];
  if(nodes.length===0){ ctx.fillText('No graph nodes',20,20); return; }
  const xs=nodes.map(n=>n.x), zs=nodes.map(n=>n.z);
  const minX=Math.min(...xs), maxX=Math.max(...xs), minZ=Math.min(...zs), maxZ=Math.max(...zs);
  const pad=20, w=(maxX-minX)||1, h=(maxZ-minZ)||1;
  const sx=(x)=>pad+(x-minX)/w*(c.width-2*pad);
  const sz=(z)=>pad+(z-minZ)/h*(c.height-2*pad);
  const pos={}; nodes.forEach(n=>pos[n.id]=[sx(n.x),sz(n.z)]);
  ctx.strokeStyle='#d1d5db'; ctx.lineWidth=1;
  edges.forEach(e=>{ if(!(e.source in pos) || !(e.target in pos)) return;
    ctx.beginPath(); ctx.moveTo(pos[e.source][0],pos[e.source][1]); ctx.lineTo(pos[e.target][0],pos[e.target][1]); ctx.stroke();
  });
  const planner = (meta && meta.planner) ? meta.planner : null;
  if(planner && planner.waypoints && planner.waypoints.length>0){
    ctx.strokeStyle='#f97316'; ctx.lineWidth=2;
    ctx.beginPath();
    planner.waypoints.forEach((p,idx)=>{ const x=p[0], z=p[1];
      if(idx===0) ctx.moveTo(sx(x),sz(z)); else ctx.lineTo(sx(x),sz(z));
    });
    ctx.stroke();
  }
  ctx.strokeStyle='#ef4444'; ctx.lineWidth=2;
  door.forEach(d=>{ const a=d[0],b=d[1]; if(!(a in pos)||!(b in pos)) return;
    ctx.beginPath(); ctx.moveTo(pos[a][0],pos[a][1]); ctx.lineTo(pos[b][0],pos[b][1]); ctx.stroke();
  });
  nodes.forEach(n=>{ if(!(n.id in pos)) return;
    ctx.fillStyle=colorForRoom(n.room_id); ctx.beginPath();
    ctx.arc(pos[n.id][0],pos[n.id][1],4,0,Math.PI*2); ctx.fill();
  });
  if(trace && trace.length>1){
    ctx.strokeStyle='#111827'; ctx.lineWidth=2; ctx.beginPath();
    trace.forEach((r,i)=>{ const px=r.pose_used?r.pose_used.x:null; const pz=r.pose_used?r.pose_used.z:null;
      if(px===null||pz===null) return;
      if(i===0) ctx.moveTo(sx(px),sz(pz)); else ctx.lineTo(sx(px),sz(pz));
    }); ctx.stroke();
  }
  const cur=trace[step]||{};
  const mark=(id,col,rad)=>{ if(!(id in pos)) return; ctx.fillStyle=col; ctx.beginPath(); ctx.arc(pos[id][0],pos[id][1],rad,0,Math.PI*2); ctx.fill(); };
  mark(cur.map_node_id,'#ef4444',6);
  mark(cur.subgoal_node_id,'#f59e0b',6);
  mark(cur.goal_node_id,'#10b981',6);
}
function updateUI(meta, trace, graph, step){
  const row=trace[step]||{};
  document.getElementById('stepText').textContent = `${step} / ${Math.max(0,trace.length-1)}`;
  document.getElementById('title').textContent = `Topo Debug: ${meta.run_id || ''}`;
  document.getElementById('runInfo').textContent = `${meta.scene_id || ''} | ${meta.query || ''} | ${meta.condition || ''}`;
  document.getElementById('termInfo').textContent = `success=${meta.success?1:0}, terminated_by=${meta.terminated_by || 'n/a'}, fail_reason=${meta.fail_reason || 'n/a'}`;
  const o=meta.oracle||{};
  document.getElementById('oracleInfo').textContent = `available=${o.oracle_available||0}, success=${o.oracle_success||0}, geo_start=${(o.oracle_geodesic_start||0).toFixed?o.oracle_geodesic_start.toFixed(3):o.oracle_geodesic_start}, geo_final=${(o.oracle_geodesic_final||0).toFixed?o.oracle_geodesic_final.toFixed(3):o.oracle_geodesic_final}`;
  const f=meta.forensics||{};
  document.getElementById('forensicsInfo').textContent = `drift=${f.cause_localization_drift?1:0}, oscillation=${f.cause_controller_oscillation?1:0}, fwd_no_motion=${f.cause_fwd_no_motion?1:0}, dataset_issue=${f.cause_dataset_reachability?1:0}`;
  const gi=meta.goal_info||{};
  document.getElementById('goalInfo').textContent = gi.goal_id ? `id=${gi.goal_id}, type=${gi.goal_type||'pose'}, pose=(${(gi.goal_pose||{}).x||'?'},${(gi.goal_pose||{}).z||'?'}), topK=${JSON.stringify(gi.topk_matches||[])}` : 'n/a (v24 run)';
  const rcs=meta.rgb_capture_stats||{};
  const rgbOk=rcs.rgb_ok||0, rgbPh=rcs.rgb_placeholder||0;
  const rgbTotal=rgbOk+rgbPh;
  const ratio = rgbTotal>0 ? (rgbPh/rgbTotal) : 0;
  const rgbOkFlag = rgbTotal>0 ? (ratio < 0.5 ? 1 : 0) : 0;
  let rgbMsg=`rgb_ok=${rgbOkFlag}, real=${rgbOk}, placeholder=${rgbPh}, placeholder_ratio=${ratio.toFixed(3)}`;
  if(rgbTotal>0 && rgbPh>rgbTotal*0.5){ rgbMsg+=' ⚠ PLACEHOLDER DOMINANT'; document.getElementById('rgbCaptureInfo').style.color='#d97706'; } else { document.getElementById('rgbCaptureInfo').style.color=''; }
  document.getElementById('rgbCaptureInfo').textContent = rgbTotal>0 ? rgbMsg : 'n/a';
  document.getElementById('rendererInfo').textContent = `${meta.renderer_used || 'unknown'}`;
  const img=document.getElementById('frame');
  img.src = row.frame_rgb ? ('../'+row.frame_rgb) : '';
  document.getElementById('telemetry').textContent = JSON.stringify({
    action: row.action, map_node: row.map_node_id, subgoal: row.subgoal_node_id, goal: row.goal_node_id,
    dist_goal: row.dist_to_goal_node, dist_subgoal: row.dist_to_subgoal, bearing: row.bearing_to_subgoal,
    loc_conf: row.loc_conf, entropy: row.belief_entropy, fwd_no_motion: row.fwd_no_motion,
    collision: row.collision, events: row.recovery_event
  }, null, 2);
  document.getElementById('beliefTopk').textContent = `belief_topk: ${JSON.stringify(row.belief_topk || [])}`;
  drawMap(graph, trace, step, meta);
}
loadAll().then(({meta,trace,graph})=>{
  const slider=document.getElementById('stepSlider');
  slider.max=Math.max(0,trace.length-1); slider.value=slider.max;
  const loc=trace.map(r=>Number(r.loc_conf||0));
  const dg=trace.map(r=>Number(r.dist_to_goal_node||0));
  const ds=trace.map(r=>Number(r.dist_to_subgoal||0));
  drawSeries('locPlot',loc,0,1,'#2563eb');
  drawSeries('goalPlot',dg,Math.min(...dg,0),Math.max(...dg,1),'#dc2626');
  drawSeries('subgoalPlot',ds,Math.min(...ds,0),Math.max(...ds,1),'#16a34a');
  const refresh=()=>updateUI(meta,trace,graph,Number(slider.value)||0);
  slider.oninput=refresh; refresh();
}).catch((e)=>{
  document.body.innerHTML = `<pre>Failed to load report: ${String(e)}</pre>`;
});
</script>
</body>
</html>
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, type=str)
    ap.add_argument("--build_root", required=True, type=str)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    meta_path = run_dir / "meta.json"
    trace_path = run_dir / "trace.jsonl"
    if not meta_path.exists() or not trace_path.exists():
        raise SystemExit(f"missing meta/trace: {run_dir}")
    meta = _load_json(meta_path, {})
    trace_rows = _load_trace_jsonl(trace_path)
    scene_id = str(meta.get("scene_id", ""))
    build_root = Path(args.build_root)
    build_dir = build_root / scene_id
    if not build_dir.exists():
        scene_key = Path(scene_id).stem
        alt = build_root / scene_key
        if alt.exists():
            build_dir = alt
    graph_payload = _build_graph_payload(build_dir)
    forensics = _derive_forensics(trace_rows, meta)
    meta["forensics"] = forensics
    report_dir = run_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    (report_dir / "trace.json").write_text(json.dumps(trace_rows, ensure_ascii=False))
    (report_dir / "graph.json").write_text(json.dumps(graph_payload, ensure_ascii=False))
    (report_dir / "index.html").write_text(HTML_TEMPLATE)


if __name__ == "__main__":
    main()
