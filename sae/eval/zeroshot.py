import torch
from torchvision.datasets import ImageFolder


def build_text_embeddings(unitok, class_names: dict, dataset: ImageFolder, device: torch.device) -> torch.Tensor:
    import open_clip
    try:
        tokenizer = open_clip.get_tokenizer("ViT-B-32")
        def _tok(texts):
            return tokenizer(texts)
    except AttributeError:
        def _tok(texts):
            return open_clip.tokenize(texts)

    idx_to_name = {v: class_names.get(k, k) for k, v in dataset.class_to_idx.items()}
    prompts = [
        f"a photo of a {idx_to_name.get(i, str(i))}"
        for i in range(len(dataset.class_to_idx))
    ]
    embs = []
    with torch.no_grad():
        for i in range(0, len(prompts), 256):
            tokens = _tok(prompts[i:i + 256]).to(device)
            embs.append(unitok.encode_text(tokens, normalize=True))
    return torch.cat(embs, dim=0)  # (n_classes, embed_dim)


def make_text_embs_fn(class_names: dict):
    def _fn(unitok, dataset, device):
        return build_text_embeddings(unitok, class_names, dataset, device)
    return _fn


def load_class_names(json_path: str) -> dict:
    import json
    with open(json_path) as f:
        raw = json.load(f)
    if isinstance(next(iter(raw.values())), list):
        return {v[0]: v[1] for v in raw.values()}
    return raw
