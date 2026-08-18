import argparse
import base64
import json
import sys
from io import BytesIO
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _encode_pil(img, thumb_px: int) -> str:
    """Resize a PIL image to thumb_px wide and return base64 JPEG string."""
    from PIL import Image
    w, h = img.size
    new_h = max(1, int(h * thumb_px / w))
    img = img.resize((thumb_px, new_h), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=72, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def load_feature_images(features_dir: Path, feature_ids: list, thumb_px: int) -> dict:
    """Mode A: load pre-saved feature_{fid:05d}_top*.png grid files."""
    from PIL import Image
    result = {}
    n_found = 0
    for fid in feature_ids:
        matches = list(features_dir.glob(f"feature_{fid:05d}_top*.png"))
        if matches:
            result[fid] = _encode_pil(Image.open(matches[0]).convert("RGB"), thumb_px)
            n_found += 1
        else:
            result[fid] = None
    print(f"  Found images for {n_found} / {len(feature_ids)} features")
    return result


def render_from_heaps(heap_results: dict, imagenet_val: str,
                      feature_ids: list, thumb_px: int, n_show: int = 4) -> dict:
    from PIL import Image, ImageDraw
    from torchvision.datasets import ImageFolder

    PATCH_GRID = 8
    PATCH_SIZE = 32

    def _draw_rect(img, patch_idx, color=(255, 60, 60), bw=3):
        row, col = patch_idx // PATCH_GRID, patch_idx % PATCH_GRID
        x0, y0 = col * PATCH_SIZE, row * PATCH_SIZE
        x1, y1 = x0 + PATCH_SIZE - 1, y0 + PATCH_SIZE - 1
        draw = ImageDraw.Draw(img)
        for o in range(bw):
            draw.rectangle([x0+o, y0+o, x1-o, y1-o], outline=color)
        return img

    dataset = ImageFolder(imagenet_val)
    cell = thumb_px
    ncols = min(n_show, 2)
    nrows = (n_show + ncols - 1) // ncols
    gap = 2

    result = {}
    n_found = 0
    for fid in feature_ids:
        entries = heap_results.get(fid, [])
        if not entries:
            result[fid] = None
            continue

        # Deduplicate by image index, keep top n_show
        seen, top = set(), []
        for val, img_idx, patch_idx in entries:
            if img_idx not in seen:
                seen.add(img_idx)
                top.append((val, img_idx, patch_idx))
            if len(top) >= n_show:
                break

        # Build patch map: img_idx → [patch_idx, ...]
        patch_map: dict = {}
        for val, img_idx, patch_idx in entries:
            patch_map.setdefault(img_idx, []).append(patch_idx)

        # Compose grid
        grid_w = ncols * cell + (ncols - 1) * gap
        grid_h = nrows * cell + (nrows - 1) * gap
        grid = Image.new("RGB", (grid_w, grid_h), (20, 20, 20))

        for i, (_, img_idx, _) in enumerate(top):
            path, _ = dataset.samples[img_idx]
            pil = Image.open(path).convert("RGB").resize((256, 256))
            for p in patch_map.get(img_idx, []):
                pil = _draw_rect(pil, p)
            pil = pil.resize((cell, cell), Image.LANCZOS)
            col_i, row_i = i % ncols, i // ncols
            x = col_i * (cell + gap)
            y = row_i * (cell + gap)
            grid.paste(pil, (x, y))

        result[fid] = _encode_pil(grid, grid_w)  # already at right size
        n_found += 1

    print(f"  Rendered thumbnails for {n_found} / {len(feature_ids)} features")
    return result



_HTML_TEMPLATE = """\
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>SAE Feature UMAP — {title}</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js" charset="utf-8"></script>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
  background: #0d0d0d; color: #ddd;
  font-family: "Fira Code", "Consolas", monospace;
  display: flex; height: 100vh; overflow: hidden;
}}
#plot {{ flex: 1; height: 100vh; min-width: 0; }}
#panel {{
  width: 300px; min-width: 300px; height: 100vh;
  overflow-y: auto; background: #111;
  border-left: 1px solid #2a2a2a;
  padding: 18px 14px; display: flex; flex-direction: column; gap: 10px;
}}
#panel h2 {{ font-size: 11px; color: #555; letter-spacing: 0.1em; text-transform: uppercase; }}
#feature-id {{ font-size: 20px; font-weight: bold; color: #fff; }}
#meta {{ font-size: 11px; color: #666; }}
#fire-freq {{ font-size: 12px; color: #888; }}
#img-wrap {{ width: 100%; }}
#feature-img {{ width: 100%; border-radius: 4px; margin-top: 4px; display: none; }}
#no-img {{ color: #444; font-size: 11px; display: none; padding: 8px 0; }}
.divider {{ border: none; border-top: 1px solid #222; margin: 4px 0; }}
#search-row {{ display: flex; gap: 6px; }}
#search-input {{
  flex: 1; background: #1a1a1a; border: 1px solid #2a2a2a;
  color: #ccc; padding: 5px 8px; border-radius: 3px;
  font-family: inherit; font-size: 12px;
}}
#search-input:focus {{ outline: none; border-color: #444; }}
#search-btn {{
  background: #1a1a1a; border: 1px solid #2a2a2a; color: #888;
  padding: 5px 10px; border-radius: 3px; cursor: pointer;
  font-family: inherit; font-size: 12px;
}}
#search-btn:hover {{ background: #252525; color: #ccc; }}
#search-msg {{ font-size: 10px; color: #555; min-height: 14px; }}
#hint {{ color: #333; font-size: 10px; margin-top: auto; line-height: 1.6; }}
.section-label {{ font-size: 10px; color: #555; letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 2px; }}
.slider-row {{ display: flex; align-items: center; gap: 6px; }}
.slider-row input[type=range] {{ flex: 1; accent-color: #7ec8e3; cursor: pointer; }}
.slider-tag {{ font-size: 10px; color: #555; width: 24px; flex-shrink: 0; }}
.slider-val {{ font-size: 10px; color: #888; width: 40px; text-align: right; flex-shrink: 0; }}
#filter-count {{ font-size: 10px; color: #555; }}
</style>
</head>
<body>
<div id="plot"></div>
<div id="panel">
  <h2>SAE Feature UMAP</h2>
  <div id="meta">{meta}</div>
  <hr class="divider">
  <div id="search-row">
    <input id="search-input" type="number" min="0" placeholder="feature ID…">
    <button id="search-btn" onclick="searchFeature()">Go</button>
  </div>
  <div id="search-msg"></div>
  <hr class="divider">
  <div id="freq-filter">
    <div class="section-label">frequency filter</div>
    <div class="slider-row">
      <span class="slider-tag">min</span>
      <input type="range" id="freq-min" min="0" max="1" step="0.001" value="0" oninput="applyFilter()">
      <span id="freq-min-val" class="slider-val">0.0%</span>
    </div>
    <div class="slider-row">
      <span class="slider-tag">max</span>
      <input type="range" id="freq-max" min="0" max="1" step="0.001" value="1" oninput="applyFilter()">
      <span id="freq-max-val" class="slider-val">100.0%</span>
    </div>
    <div id="filter-count"></div>
  </div>
  <hr class="divider">
  <div id="feature-id">hover a point</div>
  <div id="fire-freq"></div>
  <div id="img-wrap">
    <img id="feature-img" src="" alt="top activating images">
    <div id="no-img">no images for this feature<br>(run full feature analysis to populate)</div>
  </div>
  <div id="hint">
    drag → rotate · scroll → zoom<br>
    hover / search → inspect feature<br>
    color → {color_legend}
  </div>
</div>

<script>
const FDATA    = {feature_data_js};
const trace    = {trace_js};
const hl_trace = {hl_trace_js};
let _gd = null;

const ORIG_X    = trace.x;
const ORIG_Y    = trace.y;
const ORIG_Z    = trace.z;
const ORIG_FIDS = trace.customdata;
const ORIG_CLR  = Array.isArray(trace.marker.color) ? trace.marker.color : null;
const ORIG_TXT  = trace.text;

function applyFilter() {{
  const lo = parseFloat(document.getElementById('freq-min').value);
  const hi = parseFloat(document.getElementById('freq-max').value);
  document.getElementById('freq-min-val').textContent = (lo * 100).toFixed(1) + '%';
  document.getElementById('freq-max-val').textContent = (hi * 100).toFixed(1) + '%';
  const idxs = [];
  for (let i = 0; i < ORIG_FIDS.length; i++) {{
    const f = FDATA[ORIG_FIDS[i]]?.freq ?? 0;
    if (f >= lo && f <= hi) idxs.push(i);
  }}
  const pick = arr => idxs.map(i => arr[i]);
  const update = {{
    x: [pick(ORIG_X)], y: [pick(ORIG_Y)], z: [pick(ORIG_Z)],
    customdata: [pick(ORIG_FIDS)],
    text: [pick(ORIG_TXT)]
  }};
  if (ORIG_CLR) update['marker.color'] = [pick(ORIG_CLR)];
  if (_gd) Plotly.restyle(_gd, update, [0]);
  document.getElementById('filter-count').textContent =
    idxs.length + ' / ' + ORIG_FIDS.length + ' shown';
}}

function updatePanel(fid) {{
  document.getElementById('feature-id').textContent = 'Feature ' + fid;
  const fd    = FDATA[fid] || {{}};
  const imgEl = document.getElementById('feature-img');
  const noImg = document.getElementById('no-img');
  if (fd.img) {{
    imgEl.src = 'data:image/jpeg;base64,' + fd.img;
    imgEl.style.display = 'block'; noImg.style.display = 'none';
  }} else {{
    imgEl.style.display = 'none'; noImg.style.display = 'block';
  }}
  document.getElementById('fire-freq').textContent =
    fd.freq != null ? ('fires in ' + (fd.freq * 100).toFixed(1) + '% of images') : '';
}}

function searchFeature() {{
  const raw = document.getElementById('search-input').value.trim();
  const fid = parseInt(raw, 10);
  const msg = document.getElementById('search-msg');
  if (isNaN(fid) || !(fid in FDATA)) {{
    msg.textContent = raw ? 'feature ' + raw + ' not found' : '';
    return;
  }}
  msg.textContent = '';
  updatePanel(fid);
  if (!_gd) return;
  const fd = FDATA[fid];
  Plotly.restyle(_gd, {{ x: [[fd.x]], y: [[fd.y]], z: [[fd.z]] }}, [1]);
  const r = Math.sqrt(fd.x*fd.x + fd.y*fd.y + fd.z*fd.z) || 1;
  Plotly.relayout(_gd, {{ 'scene.camera': {{
    eye:    {{ x: fd.x/r*2.2, y: fd.y/r*2.2, z: fd.z/r*2.2 }},
    center: {{ x: 0, y: 0, z: 0 }},
    up:     {{ x: 0, y: 0, z: 1 }}
  }} }});
}}

document.addEventListener('DOMContentLoaded', function() {{
  document.getElementById('search-input').addEventListener('keydown', function(e) {{
    if (e.key === 'Enter') searchFeature();
  }});
}});

Plotly.newPlot('plot', [trace, hl_trace], {layout_js}, {{responsive: true, displayModeBar: true}})
  .then(function(gd) {{
    _gd = gd;
    if (!ORIG_CLR) {{
      document.getElementById('freq-filter').style.display = 'none';
    }} else {{
      document.getElementById('filter-count').textContent =
        ORIG_FIDS.length + ' / ' + ORIG_FIDS.length + ' shown';
    }}
    gd.on('plotly_hover', function(ev) {{
      updatePanel(ev.points[0].customdata);
    }});
  }});
</script>
</body>
</html>
"""


def build_html(coords: dict, feature_imgs: dict) -> str:
    fids      = coords["feature_ids"]
    xs        = coords["x"]
    ys        = coords["y"]
    zs        = coords["z"]
    fire_freq = coords.get("fire_freq")
    block     = coords.get("block", "?")
    k         = coords.get("k", "?")
    n_feat    = coords.get("n_features", len(fids))

    # Per-point color
    if fire_freq:
        colors         = fire_freq
        colorbar_title = "activation freq"
        color_legend   = "activation frequency"
    else:
        colors         = [0.5] * len(fids)
        colorbar_title = ""
        color_legend   = "uniform (no freq data)"

    # Compact feature data for JS: img, freq, and UMAP coords for search highlight
    parts = []
    for i, fid in enumerate(fids):
        img  = feature_imgs.get(fid)
        freq = fire_freq[fid] if fire_freq else None
        img_part  = f'"{img}"'  if img  else "null"
        freq_part = f"{freq:.6f}" if freq is not None else "null"
        parts.append(
            f'{fid}:{{img:{img_part},freq:{freq_part},'
            f'x:{xs[i]:.4f},y:{ys[i]:.4f},z:{zs[i]:.4f}}}'
        )
    feature_data_js = "{" + ",".join(parts) + "}"

    hl_trace = {
        "type": "scatter3d",
        "name": "selected",
        "x": [], "y": [], "z": [],
        "mode": "markers",
        "hoverinfo": "skip",
        "showlegend": False,
        "marker": {"size": 10, "color": "#facc15", "opacity": 1.0},
    }

    trace = {
        "type": "scatter3d",
        "x": xs, "y": ys, "z": zs,
        "mode": "markers",
        "text": [f"Feature {f}" for f in fids],
        "customdata": fids,
        "hovertemplate": "<b>%{text}</b><extra></extra>",
        "marker": {
            "size": 2.5,
            "color": colors,
            "colorscale": "Plasma",
            "opacity": 0.85,
            "showscale": bool(fire_freq),
            "colorbar": {"title": colorbar_title, "thickness": 12, "len": 0.6} if fire_freq else {},
        },
    }

    layout = {
        "scene": {
            "xaxis": {"visible": False},
            "yaxis": {"visible": False},
            "zaxis": {"visible": False},
            "bgcolor": "#0d0d0d",
        },
        "paper_bgcolor": "#0d0d0d",
        "margin": {"l": 0, "r": 0, "t": 0, "b": 0},
        "uirevision": "constant",
    }

    n_with_imgs = sum(1 for v in feature_imgs.values() if v is not None)
    meta = f"block={block} · k={k} · {n_feat} features · {n_with_imgs} with images"
    title = f"block{block}_k{k}"

    return _HTML_TEMPLATE.format(
        title=title,
        meta=meta,
        color_legend=color_legend,
        feature_data_js=feature_data_js,
        trace_js=json.dumps(trace),
        hl_trace_js=json.dumps(hl_trace),
        layout_js=json.dumps(layout),
    )



def main():
    p = argparse.ArgumentParser()
    p.add_argument("--umap_coords",   required=True,
                   help="JSON file from compute_feature_umap.py")
    p.add_argument("--output",        default="sae/analysis/feature_umap.html")
    p.add_argument("--thumb_px",      type=int, default=200,
                   help="Thumbnail cell size in px (use 100 for full 8192-feature build)")

    # Mode A: pre-saved PNG grids
    g_a = p.add_argument_group("Mode A — pre-saved PNG grids")
    g_a.add_argument("--features_dir", default=None,
                     help="Directory with feature_XXXXX_topN.png files")

    # Mode B: render from heap results (no PNG storage needed)
    g_b = p.add_argument_group("Mode B — render from heap results (no PNG storage)")
    g_b.add_argument("--heap_results",  default=None,
                     help="heap_results.pt saved by feature_analysis.py --skip_images")
    g_b.add_argument("--imagenet_val",  default=None,
                     help="ImageNet val root (required for Mode B)")
    g_b.add_argument("--n_show",        type=int, default=4,
                     help="Number of top images to show per feature (Mode B)")
    args = p.parse_args()

    if not args.features_dir and not args.heap_results:
        p.error("Provide either --features_dir (Mode A) or --heap_results + --imagenet_val (Mode B)")
    if args.heap_results and not args.imagenet_val:
        p.error("--imagenet_val is required when using --heap_results (Mode B)")

    try:
        from PIL import Image  # noqa
    except ImportError:
        raise ImportError("Install Pillow:  pip install Pillow")

    print(f"Loading UMAP coords from {args.umap_coords}")
    with open(args.umap_coords) as f:
        coords = json.load(f)
    n_feat = coords["n_features"]
    print(f"  {n_feat} features  block={coords.get('block','?')}  k={coords.get('k','?')}")

    if args.heap_results:
        import torch
        print(f"Mode B: rendering thumbnails from {args.heap_results} + ImageNet val ...")
        data = torch.load(args.heap_results, map_location="cpu")
        heap_results = data["results"]
        feature_imgs = render_from_heaps(
            heap_results, args.imagenet_val,
            coords["feature_ids"], args.thumb_px, args.n_show,
        )
    else:
        print(f"Mode A: loading pre-saved PNGs from {args.features_dir} ...")
        feature_imgs = load_feature_images(
            Path(args.features_dir), coords["feature_ids"], args.thumb_px
        )

    print("Building HTML ...")
    html = build_html(coords, feature_imgs)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"Saved → {out_path}  ({size_mb:.1f} MB)")
    print("Open the HTML file in any browser. Requires internet for Plotly CDN.")


if __name__ == "__main__":
    main()
