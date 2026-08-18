"""
Given a single query image, extracts all SAE features active in it, then
renders a 3D UMAP showing:
  - grey background: all 8192 features (global context)
  - colored overlay: features active in the query image, colored by
    normalized activation strength (0=low, 1=highest in this image)
  - hover panel: max-activating ImageNet images for each active feature

Usage:
    python sae/image_umap.py \\
        --image_path   path/to/query.jpg \\
        --checkpoint   sae/checkpoints/sae_block15_k64_final.pt \\
        --unitok_ckpt  unitok_tokenizer.pth \\
        --umap_coords  sae/analysis/umap_coords.json \\
        --heap_results sae/analysis/features/heap_results.pt \\
        --imagenet_val /path/to/imagenet/val \\
        [--output      sae/analysis/image_umap.html] \\
        [--block 15] [--thumb_px 120] [--n_show 4]

Usage (notebook -- models already loaded):
    from sae.image_umap import extract_image_features, build_image_umap_html
    feature_acts = extract_image_features(sae, unitok, img, block=BLOCK)
    html = build_image_umap_html(feature_acts, ...)
"""
import argparse
import base64
import json
import sys
from io import BytesIO
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


@torch.no_grad()
def extract_image_features(sae, unitok, imgs, block: int = 15) -> dict:
    """
    Pass imgs through UniTok encoder up to block `block`, then through SAE.

    Returns:
        dict mapping feature_id to max activation across all patches (and images
        in the batch). Values are strictly positive.
    """
    captured = {}

    def _hook(module, inp, out):
        B, L, D = out.shape
        acts_flat = out.detach().float().reshape(B * L, D)
        _, topk_idx, topk_vals, _ = sae.encode(acts_flat)
        captured["idx"]  = topk_idx.cpu()   # (B*L, k)
        captured["vals"] = topk_vals.cpu()  # (B*L, k)

    handle = unitok.encoder.blocks[block].register_forward_hook(_hook)
    try:
        unitok.encoder(imgs)
    finally:
        handle.remove()

    idx  = captured["idx"].numpy()   # (B*L, k)
    vals = captured["vals"].numpy()  # (B*L, k)

    feature_acts: dict = {}
    n_tok, k = idx.shape
    for t in range(n_tok):
        for j in range(k):
            fid = int(idx[t, j])
            val = float(vals[t, j])
            if val > 0.0:
                if fid not in feature_acts or val > feature_acts[fid]:
                    feature_acts[fid] = val

    return feature_acts


_PATCH_GRID = 8
_PATCH_SIZE = 32  # 256px image / 8 patches per side


def _draw_rect(img, patch_idx: int, color=(255, 60, 60), bw: int = 3):
    from PIL import ImageDraw
    row, col = patch_idx // _PATCH_GRID, patch_idx % _PATCH_GRID
    x0, y0 = col * _PATCH_SIZE, row * _PATCH_SIZE
    x1, y1 = x0 + _PATCH_SIZE - 1, y0 + _PATCH_SIZE - 1
    draw = ImageDraw.Draw(img)
    for o in range(bw):
        draw.rectangle([x0 + o, y0 + o, x1 - o, y1 - o], outline=color)
    return img


def _b64_jpeg(pil, quality: int = 72) -> str:
    buf = BytesIO()
    pil.save(buf, format="JPEG", quality=quality, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _render_thumbnails(heap_results: dict, imagenet_val: str,
                       active_fids: list, thumb_px: int, n_show: int) -> dict:
    """
    For each active feature, load its top-n_show ImageNet images from
    heap_results, draw red patch rectangles, arrange in a 2x2 grid,
    and encode as a base64 JPEG string.  Returns {fid: b64_str_or_None}.
    """
    from PIL import Image
    from torchvision.datasets import ImageFolder

    dataset = ImageFolder(imagenet_val)
    ncols   = min(n_show, 2)
    nrows   = (n_show + ncols - 1) // ncols
    gap     = 2
    grid_w  = ncols * thumb_px + (ncols - 1) * gap
    grid_h  = nrows * thumb_px + (nrows - 1) * gap

    result = {}
    for fid in active_fids:
        entries = heap_results.get(fid, [])
        if not entries:
            result[fid] = None
            continue

        # Deduplicate by image, keep top n_show distinct images
        seen, top = set(), []
        for val, img_idx, patch_idx in entries:
            if img_idx not in seen:
                seen.add(img_idx)
                top.append((val, img_idx, patch_idx))
            if len(top) >= n_show:
                break

        # All activating patches per image (for highlighting)
        patch_map: dict = {}
        for val, img_idx, patch_idx in entries:
            patch_map.setdefault(img_idx, []).append(patch_idx)

        grid = Image.new("RGB", (grid_w, grid_h), (20, 20, 20))
        for i, (_, img_idx, _) in enumerate(top):
            path, _ = dataset.samples[img_idx]
            pil = Image.open(path).convert("RGB").resize((256, 256))
            for p in patch_map.get(img_idx, []):
                pil = _draw_rect(pil, p)
            pil = pil.resize((thumb_px, thumb_px), Image.LANCZOS)
            col_i, row_i = i % ncols, i // ncols
            grid.paste(pil, (col_i * (thumb_px + gap), row_i * (thumb_px + gap)))

        result[fid] = _b64_jpeg(grid)

    return result



_HTML = """\
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Image Feature UMAP - {n_active} active features</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js" charset="utf-8"></script>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
  background: #0d0d0d; color: #ccc;
  font-family: "Fira Code", "Consolas", monospace;
  display: flex; height: {height}; overflow: hidden;
}}
#plot {{ flex: 1; height: {height}; min-width: 0; }}
#panel {{
  width: 300px; min-width: 300px; height: {height};
  overflow-y: auto; background: #111;
  border-left: 1px solid #222;
  padding: 16px 14px; display: flex; flex-direction: column; gap: 10px;
}}
#panel h2 {{ font-size: 10px; color: #444; letter-spacing: 0.1em; text-transform: uppercase; }}
#query-img {{ width: 100%; border-radius: 4px; margin-top: 4px; }}
#query-label {{ font-size: 10px; color: #444; margin-top: 4px; }}
#active-count {{ font-size: 13px; color: #666; }}
.divider {{ border: none; border-top: 1px solid #1a1a1a; margin: 4px 0; }}
#search-row {{ display: flex; gap: 6px; }}
#search-input {{
  flex: 1; background: #1a1a1a; border: 1px solid #222;
  color: #ccc; padding: 5px 8px; border-radius: 3px;
  font-family: inherit; font-size: 12px;
}}
#search-input:focus {{ outline: none; border-color: #444; }}
#search-btn {{
  background: #1a1a1a; border: 1px solid #222; color: #666;
  padding: 5px 10px; border-radius: 3px; cursor: pointer;
  font-family: inherit; font-size: 12px;
}}
#search-btn:hover {{ background: #222; color: #ccc; }}
#search-msg {{ font-size: 10px; color: #555; min-height: 14px; }}
#feature-id {{ font-size: 20px; font-weight: bold; color: #fff; }}
#act-val {{ font-size: 11px; color: #7ec8e3; }}
#feature-img {{ width: 100%; border-radius: 4px; margin-top: 4px; display: none; }}
#no-img {{ color: #333; font-size: 11px; display: none; padding: 8px 0; }}
#hint {{
  color: #2a2a2a; font-size: 10px; margin-top: auto;
  line-height: 1.7; padding-top: 8px;
}}
.section-label {{ font-size: 10px; color: #444; letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 2px; }}
.slider-row {{ display: flex; align-items: center; gap: 6px; }}
.slider-row input[type=range] {{ flex: 1; accent-color: #7ec8e3; cursor: pointer; }}
.slider-tag {{ font-size: 10px; color: #555; width: 24px; flex-shrink: 0; }}
.slider-val {{ font-size: 10px; color: #888; width: 38px; text-align: right; flex-shrink: 0; }}
#filter-count {{ font-size: 10px; color: #555; }}
</style>
</head>
<body>
<div id="plot"></div>
<div id="panel">
  <h2>Image Feature UMAP</h2>
  <img id="query-img" src="data:image/jpeg;base64,{query_b64}" alt="query">
  <div id="query-label">query image</div>
  <div id="active-count">{n_active} active features</div>
  <hr class="divider">
  <div id="search-row">
    <input id="search-input" type="number" min="0" placeholder="feature ID...">
    <button id="search-btn" onclick="searchFeature()">Go</button>
  </div>
  <div id="search-msg"></div>
  <hr class="divider">
  <div class="section-label">activation filter</div>
  <div class="slider-row">
    <span class="slider-tag">min</span>
    <input type="range" id="act-min" min="0" max="1" step="0.01" value="0" oninput="applyFilter()">
    <span id="act-min-val" class="slider-val">0.00</span>
  </div>
  <div class="slider-row">
    <span class="slider-tag">max</span>
    <input type="range" id="act-max" min="0" max="1" step="0.01" value="1" oninput="applyFilter()">
    <span id="act-max-val" class="slider-val">1.00</span>
  </div>
  <div id="filter-count"></div>
  <hr class="divider">
  <div id="feature-id">hover a colored point</div>
  <div id="act-val"></div>
  <img id="feature-img" src="" alt="top activating images">
  <div id="no-img">no images for this feature</div>
  <div id="hint">
    drag = rotate, scroll = zoom<br>
    hover colored dot / search = inspect<br>
    color = normalized activation<br>
    grey dots = all 8192 features
  </div>
</div>
<script>
const FDATA    = {feature_data_js};
const bg       = {bg_trace_js};
const active   = {active_trace_js};
const hl_trace = {hl_trace_js};
let _gd = null;

const ORIG_AX    = active.x;
const ORIG_AY    = active.y;
const ORIG_AZ    = active.z;
const ORIG_AFIDS = active.customdata;
const ORIG_ANORM = active.marker.color;
const ORIG_ATXT  = active.text;

function applyFilter() {{
  const lo = parseFloat(document.getElementById('act-min').value);
  const hi = parseFloat(document.getElementById('act-max').value);
  document.getElementById('act-min-val').textContent = lo.toFixed(2);
  document.getElementById('act-max-val').textContent = hi.toFixed(2);
  const idxs = [];
  for (let i = 0; i < ORIG_AFIDS.length; i++) {{
    const n = ORIG_ANORM[i];
    if (n >= lo && n <= hi) idxs.push(i);
  }}
  const pick = arr => idxs.map(i => arr[i]);
  const update = {{
    x: [pick(ORIG_AX)], y: [pick(ORIG_AY)], z: [pick(ORIG_AZ)],
    customdata: [pick(ORIG_AFIDS)],
    'marker.color': [pick(ORIG_ANORM)],
    text: [pick(ORIG_ATXT)]
  }};
  if (_gd) Plotly.restyle(_gd, update, [1]);
  document.getElementById('filter-count').textContent =
    idxs.length + ' / ' + ORIG_AFIDS.length + ' active shown';
}}

function updatePanel(fid) {{
  const fd = FDATA[fid] || {{}};
  document.getElementById('feature-id').textContent = 'Feature ' + fid;
  document.getElementById('act-val').textContent =
    fd.act != null
      ? 'max act: ' + fd.act.toFixed(3) + '  norm: ' + fd.norm.toFixed(3)
      : '';
  const imgEl = document.getElementById('feature-img');
  const noImg = document.getElementById('no-img');
  if (fd.img) {{
    imgEl.src = 'data:image/jpeg;base64,' + fd.img;
    imgEl.style.display = 'block'; noImg.style.display = 'none';
  }} else {{
    imgEl.style.display = 'none'; noImg.style.display = 'block';
  }}
}}

function searchFeature() {{
  const raw = document.getElementById('search-input').value.trim();
  const fid = parseInt(raw, 10);
  const msg = document.getElementById('search-msg');
  if (isNaN(fid) || !(fid in FDATA)) {{
    msg.textContent = raw ? 'feature ' + raw + ' not in this image' : '';
    return;
  }}
  msg.textContent = '';
  updatePanel(fid);
  if (!_gd) return;
  const fd = FDATA[fid];
  Plotly.restyle(_gd, {{ x: [[fd.x]], y: [[fd.y]], z: [[fd.z]] }}, [2]);
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

Plotly.newPlot('plot', [bg, active, hl_trace], {layout_js}, {{responsive: true, displayModeBar: true}})
  .then(function(gd) {{
    _gd = gd;
    document.getElementById('filter-count').textContent =
      ORIG_AFIDS.length + ' / ' + ORIG_AFIDS.length + ' active shown';
    gd.on('plotly_hover', function(ev) {{
      const pt = ev.points[0];
      if (pt.data.name !== 'active') return;
      updatePanel(pt.customdata);
    }});
    gd.on('plotly_unhover', function() {{
      document.getElementById('feature-id').textContent = 'hover a colored point';
      document.getElementById('act-val').textContent = '';
      document.getElementById('feature-img').style.display = 'none';
      document.getElementById('no-img').style.display = 'none';
    }});
  }});
</script>
</body>
</html>
"""


def build_image_umap_html(
    feature_acts: dict,
    umap_coords_path: str,
    heap_results_path: str,
    imagenet_val_path: str,
    query_img_tensor=None,
    thumb_px: int = 120,
    n_show: int = 4,
    height: str = "100vh",
) -> str:
    """
    Build a standalone HTML string showing the active SAE features of one
    image overlaid on the global UMAP.

    Args:
        feature_acts:      {feature_id: max_activation} from extract_image_features()
        umap_coords_path:  path to umap_coords.json
        heap_results_path: path to heap_results.pt
        imagenet_val_path: ImageNet val root for hover thumbnails
        query_img_tensor:  (1, 3, H, W) float in [-1, 1]; shown as thumbnail in panel
        thumb_px:          thumbnail cell width in pixels
        n_show:            number of top images shown per hovered feature

    Returns:
        Standalone HTML string (Plotly via CDN, requires internet).
    """
    from PIL import Image

    # Load global UMAP coordinates
    with open(umap_coords_path) as f:
        coords = json.load(f)

    all_fids = coords["feature_ids"]
    all_x, all_y, all_z = coords["x"], coords["y"], coords["z"]

    fid_to_pos = {fid: (all_x[i], all_y[i], all_z[i]) for i, fid in enumerate(all_fids)}

    # Filter active features to those that appear in the UMAP
    active_fids  = [f for f in feature_acts if f in fid_to_pos]
    max_act      = max(feature_acts[f] for f in active_fids) if active_fids else 1.0

    active_x     = [fid_to_pos[f][0] for f in active_fids]
    active_y     = [fid_to_pos[f][1] for f in active_fids]
    active_z     = [fid_to_pos[f][2] for f in active_fids]
    active_norms = [feature_acts[f] / max_act for f in active_fids]
    active_acts  = [feature_acts[f] for f in active_fids]

    # Render hover thumbnails (only for active features, faster)
    print("Loading heap results ...")
    heap_data    = torch.load(heap_results_path, map_location="cpu")
    heap_results = heap_data["results"]

    print(f"Rendering thumbnails for {len(active_fids)} active features ...")
    feature_imgs = _render_thumbnails(
        heap_results, imagenet_val_path, active_fids, thumb_px, n_show
    )

    # Compact JS data dict: {fid: {img, norm, act, x, y, z}}
    # x/y/z included so the search handler can move the highlight ring
    parts = []
    for fid, norm, act in zip(active_fids, active_norms, active_acts):
        img_js = f'"{feature_imgs[fid]}"' if feature_imgs.get(fid) else "null"
        x, y, z = fid_to_pos[fid]
        parts.append(
            f"{fid}:{{img:{img_js},norm:{norm:.5f},act:{act:.4f},"
            f"x:{x:.4f},y:{y:.4f},z:{z:.4f}}}"
        )
    feature_data_js = "{" + ",".join(parts) + "}"

    # Query image thumbnail (convert [-1,1] tensor to PIL)
    if query_img_tensor is not None:
        t   = query_img_tensor[0].detach().cpu().float()
        arr = t.add(1).mul(127.5).clamp(0, 255).byte().permute(1, 2, 0).numpy()
        qpil = Image.fromarray(arr).resize((200, 200), Image.LANCZOS)
    else:
        qpil = Image.new("RGB", (200, 200), (40, 40, 40))
    query_b64 = _b64_jpeg(qpil)

    # Plotly traces
    bg_trace = {
        "type": "scatter3d",
        "name": "background",
        "x": all_x, "y": all_y, "z": all_z,
        "mode": "markers",
        "hoverinfo": "skip",
        "marker": {"size": 1.5, "color": "#1a1a2e", "opacity": 0.55},
        "showlegend": False,
    }

    active_trace = {
        "type": "scatter3d",
        "name": "active",
        "x": active_x, "y": active_y, "z": active_z,
        "mode": "markers",
        "text": [f"Feature {f}" for f in active_fids],
        "customdata": active_fids,
        "hovertemplate": "<b>%{text}</b><extra></extra>",
        "marker": {
            "size": 5,
            "color": active_norms,
            "colorscale": "Viridis",
            "cmin": 0.0,
            "cmax": 1.0,
            "opacity": 0.95,
            "showscale": True,
            "colorbar": {
                "title": "norm. act.",
                "thickness": 12,
                "len": 0.55,
                "tickfont": {"color": "#666"},
                "titlefont": {"color": "#666"},
            },
        },
    }

    hl_trace = {
        "type": "scatter3d",
        "name": "selected",
        "x": [], "y": [], "z": [],
        "mode": "markers",
        "hoverinfo": "skip",
        "showlegend": False,
        "marker": {"size": 10, "color": "#facc15", "opacity": 1.0},
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

    return _HTML.format(
        n_active        = len(active_fids),
        query_b64       = query_b64,
        feature_data_js = feature_data_js,
        bg_trace_js     = json.dumps(bg_trace),
        active_trace_js = json.dumps(active_trace),
        hl_trace_js     = json.dumps(hl_trace),
        layout_js       = json.dumps(layout),
        height          = height,
    )


def main():
    p = argparse.ArgumentParser(description="Image-specific SAE feature UMAP")
    p.add_argument("--image_path",   required=True, help="Query image file")
    p.add_argument("--checkpoint",   required=True, help="SAE checkpoint .pt")
    p.add_argument("--unitok_ckpt",  required=True, help="UniTok checkpoint .pth")
    p.add_argument("--umap_coords",  required=True, help="umap_coords.json from compute_feature_umap.py")
    p.add_argument("--heap_results", required=True, help="heap_results.pt from feature_analysis.py --skip_images")
    p.add_argument("--imagenet_val", required=True, help="ImageNet val root directory")
    p.add_argument("--output",       default="sae/analysis/image_umap.html")
    p.add_argument("--block",        type=int, default=15, help="SAE hook block index")
    p.add_argument("--thumb_px",     type=int, default=120, help="Thumbnail cell size in px")
    p.add_argument("--n_show",       type=int, default=4,   help="Max images shown per hover")
    args = p.parse_args()

    try:
        from PIL import Image
    except ImportError:
        raise ImportError("pip install Pillow")

    from torchvision import transforms

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    from sae.model import TopKSAE
    from sae.unitok_loader import build_unitok

    ckpt   = torch.load(args.checkpoint, map_location="cpu")
    sae    = TopKSAE(ckpt["input_dim"], ckpt["hidden_dim"], ckpt["k"]).eval().to(device)
    sae.load_state_dict(ckpt["sae"])
    unitok = build_unitok(args.unitok_ckpt, device)
    print(f"SAE: k={ckpt['k']}  hidden={ckpt['hidden_dim']}  block={args.block}")

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2 - 1),
    ])
    img = transform(Image.open(args.image_path).convert("RGB")).unsqueeze(0).to(device)
    print(f"Query: {args.image_path}")

    feature_acts = extract_image_features(sae, unitok, img, block=args.block)
    print(f"Active features: {len(feature_acts)}  |  max act: {max(feature_acts.values()):.3f}")

    html = build_image_umap_html(
        feature_acts,
        umap_coords_path  = args.umap_coords,
        heap_results_path = args.heap_results,
        imagenet_val_path = args.imagenet_val,
        query_img_tensor  = img,
        thumb_px          = args.thumb_px,
        n_show            = args.n_show,
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"Saved -> {out}  ({out.stat().st_size / 1024 / 1024:.1f} MB)")
    print("Open in any browser (requires internet for Plotly CDN).")


if __name__ == "__main__":
    main()
