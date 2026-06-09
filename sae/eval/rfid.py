import os


def compute_rfid(tmp_dirs: dict, feature_extractor_path: str | None) -> dict:
    from utils.eval_fid import get_fid_is
    rfid: dict = {}
    for split, (d_orig, d_base, d_sae) in tmp_dirs.items():
        n = len(os.listdir(d_orig))
        print(f"\n[eval] rFID [{split}]  (n={n} images)")
        base_fid, base_isc = get_fid_is(d_orig, d_base, feature_extractor_path)
        sae_fid,  sae_isc  = get_fid_is(d_orig, d_sae,  feature_extractor_path)
        rfid[split] = {
            "baseline_fid": base_fid,
            "sae_fid":      sae_fid,
            "baseline_isc": base_isc,
            "sae_isc":      sae_isc,
        }
        df = sae_fid - base_fid
        print(f"  [{split}] FID  baseline={base_fid:.3f}  SAE={sae_fid:.3f}  Δ={df:+.3f}")
    return rfid
