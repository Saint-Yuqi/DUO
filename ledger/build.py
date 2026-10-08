#!/usr/bin/env python3
"""Build the Duo ledger: ledger/*.yaml -> docs/index.html (standalone page), docs/artifact.html (body-only, for claude.ai
Artifacts), docs/ledger.json (machine-readable), paper/tables/<table>.tex (drop-in replacements for the Overleaf tables).
Usage: python ledger/build.py            (needs pyyaml; no other dependency)
"""
import datetime as dt
import json
import pathlib
import re
import subprocess
import sys

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
LED = ROOT / "ledger"
DOCS = ROOT / "docs"
TEX = ROOT / "paper" / "tables"


def load(name):
    return yaml.safe_load((LED / f"{name}.yaml").read_text())


def git_rev():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "uncommitted"


def fmt(x, nd=2):
    if x is None:
        return "--"
    if isinstance(x, str):
        return x
    s = f"{x:.{nd}f}"
    return s


def validate(runs, ext, variants, protocols, tables):
    rid = {r["id"] for r in runs}; eid = {e["id"] for e in ext}; vid = {v["id"] for v in variants} | {"none"}
    errors = []
    for r in runs:
        if r.get("protocol") not in protocols: errors.append(f"run {r['id']}: unknown protocol {r.get('protocol')}")
        if r.get("variant") not in vid: errors.append(f"run {r['id']}: unknown variant {r.get('variant')}")
        res = r.setdefault("results", {})
        if res.get("overall") is None and res.get("clean") is not None and res.get("random") is not None:
            res["overall"] = round((res["clean"] + res["random"]) / 2, 2)
        pt = r.get("per_task")
        if pt and not (ROOT / pt).exists(): errors.append(f"run {r['id']}: per_task file missing {pt}")
    for e in ext:
        if e.get("protocol") not in protocols: errors.append(f"external {e['id']}: unknown protocol")
    for t in tables:
        rows = t.get("rows", []) + [row for g in t.get("groups", []) for row in g["rows"]]
        for row in rows:
            for k in ("run", "duo"):
                if row.get(k) and row[k] not in rid: errors.append(f"table {t['id']}: unknown run {row[k]}")
            for k in ("external", "native"):
                if row.get(k) and row[k] not in rid | eid: errors.append(f"table {t['id']}: unknown source {row[k]}")
    if errors:
        print("\n".join("ERROR " + e for e in errors)); sys.exit(1)


def cell_source(ref, runs_by, ext_by):
    if ref in runs_by:
        r = runs_by[ref]; return {"kind": "run", "id": ref, "label": r["label"], "status": r["status"], "results": r.get("results", {})}
    e = ext_by[ref]; return {"kind": "external", "id": ref, "label": e["label"], "source": e["source"], "snapshot": str(e["snapshot"]), "results": e["results"]}


def resolve_tables(tables, runs_by, ext_by):
    out = []
    for t in tables:
        tt = {k: v for k, v in t.items() if k not in ("rows", "groups")}
        groups = t.get("groups") or [{"name": None, "rows": t["rows"]}]
        tt["groups"] = []
        for g in groups:
            rows = []
            for row in g["rows"]:
                rr = {"label": row["label"], "note": row.get("note"), "bold": row.get("bold", False)}
                if t["id"] == "action_head_transfer":
                    rr["native"] = cell_source(row["native"], runs_by, ext_by) if row.get("native") else None
                    rr["duo"] = cell_source(row["duo"], runs_by, ext_by) if row.get("duo") else None
                    rr["planned"] = bool(row.get("planned")); rr["planned_duo"] = bool(row.get("planned_duo")); rr["planned_native"] = bool(row.get("planned_native"))
                    n, d = (rr["native"] or {}).get("results", {}).get("overall"), (rr["duo"] or {}).get("results", {}).get("overall")
                    rr["delta"] = round(d - n, 2) if (n is not None and d is not None) else None
                else:
                    ref = row.get("run") or row.get("external")
                    rr["source"] = cell_source(ref, runs_by, ext_by) if ref else None
                    rr["planned"] = bool(row.get("planned")) or ref is None
                rows.append(rr)
            tt["groups"].append({"name": g.get("name"), "rows": rows})
        out.append(tt)
    return out


def tex_escape(s):
    return s.replace("π", r"$\pi$").replace("×", r"$\times$").replace("%", r"\%").replace("_", r"\_").replace("&", r"\&").replace("—", "--").replace("→", r"$\rightarrow$").replace("≥", r"$\geq$")


def tex_table(t):
    """LaTeX in the shape of the Overleaf table files; numbers come from the ledger."""
    cols = t["columns"]
    if t["id"] == "action_head_transfer":
        head = r"Base policy & Native overall & Duo overall & Duo clean & Duo random. & $\Delta$ overall & Status \\"
        spec = "lcccccc"; env = "table*"; width = r"\textwidth"
    else:
        names = {"overall": "Overall", "clean": "Clean", "random": "Random."}
        head = "Method & " + " & ".join(names[c] for c in cols) + r" \\"
        spec = "l" + "c" * len(cols); env = "table" if len(cols) <= 3 else "table*"; width = r"\columnwidth" if env == "table" else r"\textwidth"
    L = [f"\\begin{{{env}}}[t]", f"  \\caption{{{tex_escape(t['caption'])} Generated from ledger/ ({git_rev()}, {dt.date.today()}); edit the YAML, not this file.}}",
         f"  \\label{{tab:{t['id']}}}", "  \\centering", "  \\footnotesize", "  \\setlength{\\tabcolsep}{3pt}", "  \\renewcommand{\\arraystretch}{1.08}",
         f"  \\begin{{tabular*}}{{{width}}}{{@{{\\extracolsep{{\\fill}}}}{spec}@{{}}}}", "    \\toprule", "    " + head, "    \\midrule"]
    ncol = len(spec)
    for gi, g in enumerate(t["groups"]):
        if g["name"]:
            if gi: L.append("    \\addlinespace[2pt]")
            L.append(f"    \\multicolumn{{{ncol}}}{{@{{}}l}}{{\\textit{{{tex_escape(g['name'])}}}}} \\\\")
        for r in g["rows"]:
            lab = tex_escape(r["label"]); lab = f"\\textbf{{{lab}}}" if r.get("bold") else lab
            if t["id"] == "action_head_transfer":
                n = (r["native"] or {}).get("results", {}); d = (r["duo"] or {}).get("results", {})
                st = "Completed" if r["duo"] and r["duo"]["kind"] == "run" and r["duo"]["status"] == "complete" else ("Evaluating" if r["duo"] and r["duo"].get("status") == "evaluating" else "Planned")
                vals = [fmt(n.get("overall")), fmt(d.get("overall")), fmt(d.get("clean")), fmt(d.get("random")), ("+" if (r["delta"] or 0) > 0 else "") + fmt(r["delta"]), st]
            else:
                res = (r["source"] or {}).get("results", {})
                vals = [fmt(res.get(c)) for c in cols]
            L.append(f"    {lab} & " + " & ".join(vals) + r" \\")
    L += ["    \\bottomrule", "  \\end{tabular*}", f"\\end{{{env}}}", ""]
    return "\n".join(L)


def main():
    runs, ext, variants, protocols, tables = load("runs")["runs"], load("external")["external"], load("variants")["variants"], load("protocols")["protocols"], load("tables")["tables"]
    validate(runs, ext, variants, protocols, tables)
    runs_by = {r["id"]: r for r in runs}; ext_by = {e["id"]: e for e in ext}
    per_task = {}
    for r in runs:
        if r.get("per_task") and (ROOT / r["per_task"]).exists():
            per_task[r["id"]] = json.loads((ROOT / r["per_task"]).read_text())
    rtables = resolve_tables(tables, runs_by, ext_by)
    tex = {t["id"]: tex_table(t) for t in rtables if t.get("tex")}
    TEX.mkdir(parents=True, exist_ok=True)
    for tid, s in tex.items():
        (TEX / f"{tid}.tex").write_text(s)
    build = {"time": dt.datetime.now().astimezone().isoformat(timespec="minutes"), "git": git_rev(),
             "counts": {s: sum(1 for r in runs if r["status"] == s) for s in ("complete", "evaluating", "training", "partial", "aborted", "planned", "superseded")}}
    data = {"build": build, "protocols": protocols, "runs": runs, "external": ext, "variants": variants, "tables": rtables, "tex": tex, "per_task": per_task,
            "links": {"github": "https://github.com/Saint-Yuqi/DUO", "codeup": "https://codeup.aliyun.com/6a3ce6c6a6fcee143fa25a90/DUO", "paper_mirror": "https://github.com/pikonguwu/Duo-WAM",
                      "leaderboard": "https://robotwin-platform.github.io/leaderboard", "artifact": "https://claude.ai/artifact/LYUcUNwvGWM3XQeWZgQHSR"}}
    DOCS.mkdir(exist_ok=True)
    js = json.dumps(data, ensure_ascii=False, default=str).replace("</", "<\\/")
    (DOCS / "ledger.json").write_text(json.dumps(data, ensure_ascii=False, indent=1, default=str))
    tpl = (LED / "template.html").read_text()
    body = tpl.replace("/*__LEDGER_DATA__*/", "window.LEDGER = " + js + ";")
    (DOCS / "artifact.html").write_text(body)
    (DOCS / "index.html").write_text('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">\n</head>\n<body style="margin:0">\n' + body + "\n</body>\n</html>\n")
    print(f"ledger: {len(runs)} runs, {len(ext)} external, {len(variants)} variants, {len(rtables)} tables, {len(tex)} tex files -> docs/index.html, docs/artifact.html, docs/ledger.json, paper/tables/")


if __name__ == "__main__":
    main()
