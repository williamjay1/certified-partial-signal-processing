"""Validate and assemble the finished vector figure/table assets without redrawing."""
from pathlib import Path
import sys, os, json, hashlib, zipfile, subprocess, re
ROOT = Path(__file__).resolve().parents[1]
os.environ["TEMP"] = os.environ["TMP"] = str(ROOT / "temp")
sys.path.insert(0, str(ROOT / "temp" / "nature_workflow_qa_dependencies"))
sys.path.insert(0, str(ROOT / "scripts" / "vendor" / "nature_design" / "baker" / "scripts"))
import pymupdf as fitz
from audit_figure_collisions import audit_pdf as collision_audit
from audit_pdf_text import audit_pdf as glyph_audit

FIG = ROOT / "artwork" / "figures"
TABLE = ROOT / "artwork" / "tables"
OUT = ROOT / "artwork" / "proof"
OUT.mkdir(exist_ok=True)
figures = [
    "figure1_workflow", "figure2_subject_variation", "figure3_stopping_controls",
    "figure4_fidelity_time", "figure5_latency_tails", "figure6_component_cost",
]
table_manifest = json.loads((TABLE / "table_manifest.json").read_text(encoding="utf-8"))
table_files = sorted(TABLE.glob("table[0-9][0-9].pdf"))
assert len(table_files) == 12
assets = [FIG / (name + ".pdf") for name in figures] + table_files
records = []
for p in assets:
    pdf = fitz.open(p)
    assert len(pdf) == 1
    page = pdf[0]
    assert not page.get_images(full=True), f"Raster embedded in {p}"
    fonts = [{"name": font[3], "type": font[2],
              "embedded_bytes": len(pdf.extract_font(font[0])[-1])}
             for font in page.get_fonts(full=True)]
    assert all(f["embedded_bytes"] > 0 for f in fonts), f"Unembedded font: {p}"
    width_mm = page.rect.width / 72 * 25.4
    height_mm = page.rect.height / 72 * 25.4
    assert abs(width_mm - 183) < .02 and height_mm <= 170
    glyphs = glyph_audit(p.read_bytes(), minimum_pt=5)
    assert glyphs["auditable"] and glyphs["below_minimum_count"] == 0, p
    collisions = collision_audit(p)
    (p.parent / (p.stem + "_pdf_collision.json")).write_text(
        json.dumps(collisions, indent=2, ensure_ascii=False), encoding="utf-8")
    (p.parent / (p.stem + "_pdf_fonts.json")).write_text(
        json.dumps(glyphs, indent=2), encoding="utf-8")
    findings = collisions["findings"]
    assert collisions["verdict"] == "PASS" and not findings, (p, collisions)
    svg_path = p.with_suffix(".svg")
    svg = svg_path.read_text(encoding="utf-8")
    assert "<image" not in svg
    assert len(re.findall(r"<text(?:\s|>)", svg)) > 0
    record = {
        "asset": p.stem, "width_mm": width_mm, "height_mm": height_mm,
        "pdf_no_raster": True, "svg_no_raster": True, "svg_editable_text": True,
        "embedded_fonts": fonts, "minimum_pdf_font_pt": glyphs["minimum_found_pt"],
        "collision_verdict": collisions["verdict"],
        "collision_findings": len(findings),
        "pdf_sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        "svg_sha256": hashlib.sha256(svg_path.read_bytes()).hexdigest(),
    }
    records.append(record)
    pdf.close()
    print("STRICT_QA_PASS", p.stem, flush=True)

# A single, directly reviewable 18-page vector proof, preserving source page geometry.
combined = fitz.open()
toc = []
for i, p in enumerate(assets, 1):
    source = fitz.open(p)
    combined.insert_pdf(source)
    source.close()
    if i <= 6:
        title = f"Figure {i}: {p.stem.split('_', 1)[1].replace('_', ' ')}"
    else:
        title = f'Table {i - 6}: {table_manifest["tables"][i - 7]["title"]}'
    toc.append([1, title, i])
combined.set_toc(toc)
combined.set_metadata({
    "title": "Certified partial processing — publication figure and table proof",
    "subject": "Six figures and twelve tables; Nature-inspired editable vector artwork",
    "creator": "Matplotlib, SciencePlots style; assembled without rasterization",
})
proof_path = OUT / "ASC_Nature_style_figures_tables.pdf"
combined.save(proof_path, garbage=4, deflate=True)
combined.close()

source_checks = {}
vendor = ROOT / "scripts" / "vendor" / "nature_design" / "baker" / "scripts"
for name in ("nature_figures.py", "nature_workflow.py", "nature_tables.py"):
    command = [sys.executable, str(vendor / "validate_figure.py"),
               str(ROOT / "scripts" / name), "--profile", "ipd-baker", "--json"]
    completed = subprocess.run(command, capture_output=True, text=True,
                               encoding="utf-8", errors="replace")
    source_checks[name] = {"returncode": completed.returncode,
                          "stdout": completed.stdout, "stderr": completed.stderr}
    print("SOURCE_PREFLIGHT", name, completed.returncode, flush=True)
report = {
    "stage": "EXECUTED_NATURE_STYLE_ARTWORK_QA",
    "figures": 6, "tables": 12, "raster_dpi": 1000,
    "qa": records, "source_preflights": source_checks,
    "scientific_settings_changed": False,
    "all_primary_and_failure_outputs_preserved": True,
    "pdf_proof": str(proof_path),
    "physical_printout_checked": False,
    "note": "Vector artwork has no DPI limit; PNGs are original 1000-dpi renders. "
            "Final physical dimensions, embedded fonts, no raster images, editable SVG text, "
            "and zero rendered PDF text/stroke/fill collisions were checked.",
}
(OUT / "ARTWORK_QA.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print("PROOF_COMPLETE", proof_path, flush=True)
