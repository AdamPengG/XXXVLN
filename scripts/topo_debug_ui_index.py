#!/usr/bin/env python
import argparse
import json
from pathlib import Path
from typing import Dict, List


def _load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


HTML = """<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Topo Debug Index</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 0; background: #f7f8fa; }
    .top { padding: 12px 16px; background: #111827; color: #fff; }
    .wrap { padding: 12px; }
    table { border-collapse: collapse; width: 100%; background: #fff; }
    th, td { border: 1px solid #ddd; padding: 8px; font-size: 13px; }
    th { background: #f3f4f6; text-align: left; }
    input, select { margin-right: 8px; }
  </style>
</head>
<body>
  <div class="top">Topo Debug Runs</div>
  <div class="wrap">
    <div>
      <label>condition</label><select id="fCond"><option value="">all</option></select>
      <label>scene</label><select id="fScene"><option value="">all</option></select>
      <label>terminated_by</label><select id="fTerm"><option value="">all</option></select>
      <label>fail_reason</label><select id="fFail"><option value="">all</option></select>
      <label>oracle_success</label><select id="fOracle"><option value="">all</option><option value="1">1</option><option value="0">0</option></select>
    </div>
    <table id="tbl">
      <thead>
        <tr><th>run_id</th><th>condition</th><th>scene</th><th>query</th><th>success</th><th>terminated_by</th><th>fail_reason</th><th>oracle_success</th><th>report</th></tr>
      </thead>
      <tbody></tbody>
    </table>
  </div>
<script>
const rows = __ROWS__;
function uniq(k){ return [...new Set(rows.map(r=>r[k]||''))].filter(Boolean).sort(); }
function fillSelect(id, vals){ const s=document.getElementById(id); vals.forEach(v=>{ const o=document.createElement('option'); o.value=v; o.textContent=v; s.appendChild(o); }); }
function render(){
  const cond=document.getElementById('fCond').value;
  const scene=document.getElementById('fScene').value;
  const term=document.getElementById('fTerm').value;
  const fail=document.getElementById('fFail').value;
  const ora=document.getElementById('fOracle').value;
  const tb=document.querySelector('#tbl tbody');
  tb.innerHTML='';
  rows.filter(r=>{
    if(cond && r.condition!==cond) return false;
    if(scene && r.scene_id!==scene) return false;
    if(term && r.terminated_by!==term) return false;
    if(fail && (r.fail_reason||'')!==fail) return false;
    if(ora && String(r.oracle_success)!==ora) return false;
    return true;
  }).forEach(r=>{
    const tr=document.createElement('tr');
    const cells=[r.run_id,r.condition,r.scene_id,r.query,r.success,r.terminated_by||'',r.fail_reason||'',r.oracle_success];
    cells.forEach(c=>{ const td=document.createElement('td'); td.textContent=String(c); tr.appendChild(td); });
    const td=document.createElement('td'); const a=document.createElement('a'); a.href=r.report_rel; a.textContent='open'; td.appendChild(a); tr.appendChild(td);
    tb.appendChild(tr);
  });
}
fillSelect('fCond', uniq('condition'));
fillSelect('fScene', uniq('scene_id'));
fillSelect('fTerm', uniq('terminated_by'));
fillSelect('fFail', uniq('fail_reason'));
['fCond','fScene','fTerm','fFail','fOracle'].forEach(id=>document.getElementById(id).onchange=render);
render();
</script>
</body>
</html>
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=str)
    args = ap.parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    rows: List[Dict] = []
    for meta_path in sorted(root.glob("**/meta.json")):
        run_dir = meta_path.parent
        report = run_dir / "report" / "index.html"
        if not report.exists():
            continue
        meta = _load_json(meta_path, {})
        rows.append(
            {
                "run_id": str(meta.get("run_id", run_dir.name)),
                "condition": str(meta.get("condition", run_dir.parent.name)),
                "scene_id": str(meta.get("scene_id", "")),
                "query": str(meta.get("query", "")),
                "success": int(bool(meta.get("success", False))),
                "terminated_by": meta.get("terminated_by"),
                "fail_reason": meta.get("fail_reason"),
                "oracle_success": int(meta.get("oracle", {}).get("oracle_success", 0))
                if isinstance(meta.get("oracle"), dict)
                else 0,
                "report_rel": str(report.relative_to(root)),
            }
        )
    page = HTML.replace("__ROWS__", json.dumps(rows, ensure_ascii=False))
    (root / "index.html").write_text(page)


if __name__ == "__main__":
    main()
