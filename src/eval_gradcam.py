import os
import sys
import json
import torch
import numpy as np
import open_clip
import importlib.util
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from dataset import get_splits, CLASS_NAMES, NUM_CLASSES, MEAN, STD

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "eval_gradcam.json"

def _load_config():
    with open(CONFIG_PATH) as f:
        return json.load(f)

_cfg = _load_config()

IMAGE_DIR = PROJECT_ROOT / _cfg["paths"]["image_dir"]
MASK_DIR = PROJECT_ROOT / _cfg["paths"]["mask_dir"]
OUTPUT_DIR = PROJECT_ROOT / _cfg["paths"]["output_dir"]
IMAGE_SIZE = _cfg["eval"]["image_size"]
THRESHOLD = _cfg["eval"]["threshold"]
MINORITY_CLASSES = set(_cfg["eval"]["minority_classes"])
SAMPLES_PER_FIGURE = _cfg["eval"]["samples_per_figure"]
HAM_CLASSNAMES = _cfg["class_names"]
MODEL_NAMES = ["CNN", "ResNet-50", "Swin-T", "CLIP", "DermLIP"]

CKPTS = {name: PROJECT_ROOT / path for name, path in _cfg["checkpoints"].items()}
VLM_SUMMARIES = {name: PROJECT_ROOT / path for name, path in _cfg["vlm_summaries"].items()}
MODEL_FILES = {name: PROJECT_ROOT / path for name, path in _cfg["model_files"].items()}
VLM_CFGS = {
    name: (cfg["model_id"], cfg["pretrained"])
    for name, cfg in _cfg["vlm_models"].items()
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_mask(img_id):
    path = MASK_DIR / f"{img_id}_segmentation.png"
    if not path.exists():
        return None
    mask = Image.open(path).convert("L")
    mask = mask.resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST)
    return (np.array(mask) > 128).astype(np.uint8)


def iou_dice(cam, mask):
    h_bin = (cam >= THRESHOLD).astype(np.uint8)
    inter = (h_bin & mask).sum()
    union = (h_bin | mask).sum()
    iou = inter / (union + 1e-8)
    dice = 2 * inter / (h_bin.sum() + mask.sum() + 1e-8)
    return float(iou), float(dice)


def load_image_raw(img_id):
    return Image.open(IMAGE_DIR / f"{img_id}.jpg").convert("RGB")


def img_to_np(pil_img):
    return np.array(pil_img.resize((IMAGE_SIZE, IMAGE_SIZE))) / 255.0


# ── Grad-CAM (supervised models only) ────────────────────────────────────────
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self._fwd)
        target_layer.register_full_backward_hook(self._bwd)

    def _fwd(self, _, __, output):
        self.activations = (output[0] if isinstance(output, tuple) else output).detach()

    def _bwd(self, _, __, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, x, class_idx):
        self.model.eval()
        x = x.requires_grad_(True)
        output = self.model(x)
        self.model.zero_grad()
        output[0, class_idx].backward()
        return self._compute_cam()

    def _compute_cam(self):
        acts  = self.activations
        grads = self.gradients
        if acts is None or grads is None:
            return np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)

        if acts.dim() == 3:
            n_tokens = acts.shape[1]
            h = w = int(n_tokens ** 0.5)
            if h * w != n_tokens:
                # CLS token present — skip it
                patch_acts  = acts[:, 1:, :]
                patch_grads = grads[:, 1:, :]
                h = w = int(patch_acts.shape[1] ** 0.5)
                patch_acts  = patch_acts.reshape(1, h, w, -1).permute(0, 3, 1, 2)
                patch_grads = patch_grads.reshape(1, h, w, -1).permute(0, 3, 1, 2)
                weights = patch_grads.abs().mean(dim=(2, 3), keepdim=True)
                cam = (weights * patch_acts.abs()).sum(dim=1, keepdim=True)
            else:
                # No CLS token (Swin)
                patch_acts  = acts.reshape(1, h, w, -1).permute(0, 3, 1, 2)
                patch_grads = grads.reshape(1, h, w, -1).permute(0, 3, 1, 2)
                weights = patch_grads.abs().mean(dim=(2, 3), keepdim=True)
                cam = (weights * patch_acts.abs()).sum(dim=1, keepdim=True)
        else:
            # CNN: standard Grad-CAM with ReLU
            weights = grads.mean(dim=(2, 3), keepdim=True)
            cam = F.relu((weights * acts).sum(dim=1, keepdim=True))

        cam = F.interpolate(cam, size=(IMAGE_SIZE, IMAGE_SIZE),
                            mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


# ── VLM attention visualization ───────────────────────────────────────────────
def compute_cam_vlm(vlm_model, txt_feats, preprocess, img_id, pred_cls):
    """
    Extract CLS-to-patch self-attention from the final transformer resblock.
    Recomputes attention weights from Q and K via a forward hook on
    nn.MultiheadAttention, averages over heads, and reshapes to a spatial grid.
    """
    img = preprocess(load_image_raw(img_id)).unsqueeze(0).to(device)
    last_block = vlm_model.visual.transformer.resblocks[-1]
    captured = {}

    def hook_fn(module, input, output):
        x = input[0]  # open_clip passes (batch, seq_len, embed_dim)
        
        # handle both possible layouts
        if x.shape[0] == 1:
            # (batch=1, seq_len=197, embed_dim=768) — standard layout
            batch, seq_len, embed_dim = x.shape
        else:
            # (seq_len=197, batch=1, embed_dim=768) — transposed layout
            seq_len, batch, embed_dim = x.shape
            x = x.permute(1, 0, 2)  # normalize to (batch, seq_len, embed_dim)
            batch, seq_len, embed_dim = x.shape
    
        n_heads  = module.num_heads
        head_dim = embed_dim // n_heads
    
        # in_proj_weight: (3*embed_dim, embed_dim)
        qkv = F.linear(x, module.in_proj_weight, module.in_proj_bias)
        q, k, _ = qkv.chunk(3, dim=-1)  # each (batch, seq_len, embed_dim)
    
        # reshape to (batch, n_heads, seq_len, head_dim)
        q = q.reshape(batch, seq_len, n_heads, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(batch, seq_len, n_heads, head_dim).permute(0, 2, 1, 3)
    
        scale = head_dim ** -0.5
        attn  = (q @ k.transpose(-2, -1)) * scale  # (batch, n_heads, seq_len, seq_len)
        attn  = attn.softmax(dim=-1)
    
        # CLS at position 0, patch tokens at 1:
        cls_attn = attn[0, :, 0, 1:]  # (n_heads, n_patches)
        n_patches = cls_attn.shape[1]
        h = int(n_patches ** 0.5)
        
        if h * h == n_patches:
            captured["attn"] = cls_attn.mean(dim=0).detach()
        else:
            print(f"WARNING: n_patches={n_patches} not a perfect square, skipping.")

    handle = last_block.attn.register_forward_hook(hook_fn)

    with torch.no_grad():
        _ = vlm_model.encode_image(img)

    handle.remove()

    if "attn" not in captured:
        print("WARNING: attention not captured — returning zeros.")
        return np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)

    attn_map = captured["attn"]
    h = w = int(attn_map.shape[0] ** 0.5)
    attn_map = attn_map.reshape(1, 1, h, w).float()
    attn_map = F.interpolate(attn_map, size=(IMAGE_SIZE, IMAGE_SIZE),
                             mode="bilinear", align_corners=False)
    attn_map = attn_map.squeeze().cpu().numpy()
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
    return attn_map


# ── Build all models ──────────────────────────────────────────────────────────
def _resolve_layer(model, layer_str):
    obj = model
    for part in layer_str.split("."):
        if "[" in part:
            attr, idx = part[:-1].split("[")
            obj = getattr(obj, attr)[int(idx)]
        else:
            obj = getattr(obj, part)
    return obj


def build_all_models():
    cnn_mod = load_mod("custom_cnn", MODEL_FILES["CNN"])
    resnet_mod = load_mod("resnet50", MODEL_FILES["ResNet-50"])
    swin_mod = load_mod("swin_t", MODEL_FILES["Swin-T"])

    cnn = cnn_mod.custom_CNN(num_classes=NUM_CLASSES)
    cnn.load_state_dict(torch.load(CKPTS["CNN"], map_location=device))
    cnn.to(device).eval()

    resnet = resnet_mod.build_resnet50(num_classes=NUM_CLASSES)
    resnet.load_state_dict(torch.load(CKPTS["ResNet-50"], map_location=device))
    resnet.to(device).eval()

    swin = swin_mod.build_swin_tiny(num_classes=NUM_CLASSES)
    swin.load_state_dict(torch.load(CKPTS["Swin-T"], map_location=device))
    swin.to(device).eval()

    sup_models   = {"CNN": cnn, "ResNet-50": resnet, "Swin-T": swin}
    sup_gradcams = {
        name: GradCAM(model, _resolve_layer(model, _cfg["target_layers"][name]))
        for name, model in sup_models.items()
    }

    # VLMs — no GradCAM hooks; attention extracted in compute_cam_vlm
    vlm_models, vlm_preprocesses, vlm_tokenizers, vlm_templates = {}, {}, {}, {}

    for name, (model_id, pretrained) in VLM_CFGS.items():
        print(f"  Loading {name}...")
        if pretrained:
            m, _, preprocess = open_clip.create_model_and_transforms(
                model_id, pretrained=pretrained)
        else:
            m, _, preprocess = open_clip.create_model_and_transforms(model_id)
        tokenizer = open_clip.get_tokenizer(model_id)
        m.to(device).eval()
        with open(VLM_SUMMARIES[name]) as f:
            summary = json.load(f)
        vlm_templates[name] = summary["best_config"]["template_str"]
        vlm_models[name] = m
        vlm_preprocesses[name] = preprocess
        vlm_tokenizers[name] = tokenizer

    vlm_txt_feats = {}
    for name in ["CLIP", "DermLIP"]:
        tmpl = vlm_templates[name]
        texts = vlm_tokenizers[name](
            [tmpl.format(c=c) for c in HAM_CLASSNAMES]
        ).to(device)
        with torch.no_grad():
            tf = vlm_models[name].encode_text(texts)
            tf /= tf.norm(dim=-1, keepdim=True)
        vlm_txt_feats[name] = tf

    return (sup_models, sup_gradcams,
            vlm_models, vlm_preprocesses, vlm_txt_feats)


# ── Per-sample inference ──────────────────────────────────────────────────────
def get_sup_pred(model, img_tensor):
    with torch.no_grad():
        return model(img_tensor.unsqueeze(0).to(device)).argmax(dim=1).item()


def get_vlm_pred(vlm_model, preprocess, txt_feats, img_id):
    img = preprocess(load_image_raw(img_id)).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = vlm_model.encode_image(img)
        feat /= feat.norm(dim=-1, keepdim=True)
    return (feat @ txt_feats.T).argmax(dim=1).item()


def compute_cam_sup(gradcam, img_tensor, pred_cls):
    return gradcam(img_tensor.unsqueeze(0).to(device), pred_cls)


# ── Figure generation ─────────────────────────────────────────────────────────
def make_report_figure(samples, title, filename, model_names):
    n_samples = len(samples)
    n_cols    = 1 + len(model_names)
    fig = plt.figure(figsize=(3.5 * n_cols, 5 * n_samples))
    gs  = gridspec.GridSpec(n_samples * 2, n_cols, figure=fig,
                            hspace=0.05, wspace=0.05)

    for s_idx, sample in enumerate(samples):
        row_top = s_idx * 2
        row_bot = s_idx * 2 + 1

        ax_orig = fig.add_subplot(gs[row_top:row_bot + 1, 0])
        ax_orig.imshow(sample["img_np"])
        label = f"{'Original' + chr(10) if s_idx == 0 else ''}True: {sample['true_cls']}"
        ax_orig.set_title(label, fontsize=8, pad=3)
        ax_orig.axis("off")

        for m_idx, model_name in enumerate(model_names):
            col = m_idx + 1
            cam = sample["cams"][model_name]
            pred = sample["preds"][model_name]
            mask = sample["mask"]
            color = "green" if pred == sample["true_cls"] else "red"

            ax_cam = fig.add_subplot(gs[row_top, col])
            ax_cam.imshow(sample["img_np"])
            ax_cam.imshow(cam, cmap="jet", alpha=0.45)
            header = f"{model_name}\n" if s_idx == 0 else ""
            ax_cam.set_title(f"{header}Pred: {pred}", fontsize=7,
                             color=color, pad=2)
            ax_cam.axis("off")

            ax_mask = fig.add_subplot(gs[row_bot, col])
            if mask is not None:
                iou, dice = iou_dice(cam, mask)
                ax_mask.imshow(mask, cmap="gray")
                ax_mask.set_title(f"IoU={iou:.2f} Dice={dice:.2f}",
                                  fontsize=6, pad=2)
            else:
                ax_mask.set_title("no mask", fontsize=6, pad=2)
            ax_mask.axis("off")

    fig.suptitle(title, fontsize=13, y=1.01)
    out_path = OUTPUT_DIR / "figures" / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# ── Per-model example figure ──────────────────────────────────────────────────
def save_per_model_examples(model_name, samples_by_class, model_dir):
    classes = [cls for cls in CLASS_NAMES if cls in samples_by_class]
    n = len(classes)
    if n == 0:
        return

    fig, axes = plt.subplots(3, n, figsize=(3.5 * n, 10))
    if n == 1:
        axes = axes.reshape(3, 1)

    for col, cls in enumerate(classes):
        s = samples_by_class[cls]
        cam = s["cam"]
        mask = s["mask"]
        pred = s["pred"]
        color = "green" if pred == cls else "red"

        axes[0, col].imshow(s["img_np"])
        axes[0, col].set_title(cls, fontsize=8, pad=2)
        axes[0, col].axis("off")

        axes[1, col].imshow(s["img_np"])
        axes[1, col].imshow(cam, cmap="jet", alpha=0.45)
        axes[1, col].set_title(f"Pred: {pred}", fontsize=7, color=color, pad=2)
        axes[1, col].axis("off")

        if mask is not None:
            iou, dice = iou_dice(cam, mask)
            axes[2, col].imshow(mask, cmap="gray")
            axes[2, col].set_title(f"IoU={iou:.2f}\nDice={dice:.2f}",
                                   fontsize=6, pad=2)
        else:
            axes[2, col].set_title("no mask", fontsize=6, pad=2)
        axes[2, col].axis("off")

    for row, label in enumerate(["Original", "Attn/Grad-CAM", "GT Mask"]):
        axes[row, 0].set_ylabel(label, fontsize=9, rotation=90, labelpad=4)

    fig.suptitle(f"{model_name} — 1 Example per Class", fontsize=12)
    plt.tight_layout()
    out_path = model_dir / "gradcam_examples.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# ── Collect cross-model samples ───────────────────────────────────────────────
def collect_samples(test_df, sup_models, sup_gradcams,
                    vlm_models, vlm_preprocesses, vlm_txt_feats):
    all_correct, disagreement, minority_fail = [], [], []

    print("\nCollecting cross-model samples...")
    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        img_id = row["image"]
        true_label = int(row[CLASS_NAMES].values.argmax())
        true_cls = CLASS_NAMES[true_label]
        mask = load_mask(img_id)
        if mask is None:
            continue

        img_raw = load_image_raw(img_id)
        img_np = img_to_np(img_raw)
        img_tensor = val_transform(img_raw)

        preds, cams = {}, {}

        for name, model in sup_models.items():
            pred = get_sup_pred(model, img_tensor)
            preds[name] = CLASS_NAMES[pred]
            cams[name] = compute_cam_sup(sup_gradcams[name], img_tensor, pred)

        for name in ["CLIP", "DermLIP"]:
            pred = get_vlm_pred(vlm_models[name], vlm_preprocesses[name],
                                       vlm_txt_feats[name], img_id)
            preds[name] = CLASS_NAMES[pred]
            cams[name] = compute_cam_vlm(vlm_models[name],
                                          vlm_txt_feats[name],
                                          vlm_preprocesses[name],
                                          img_id, pred)

        sample = {"img_id": img_id, "img_np": img_np, "mask": mask,
                  "true_cls": true_cls, "preds": preds, "cams": cams}

        all_pred_correct = all(preds[m] == true_cls for m in MODEL_NAMES)
        sup_preds = [preds[m] for m in ["CNN", "ResNet-50", "Swin-T"]]
        all_disagree = len(set(sup_preds)) == 3
        sup_all_wrong = all(preds[m] != true_cls for m in ["CNN", "ResNet-50", "Swin-T"])
        vlm_all_wrong = all(preds[m] != true_cls for m in ["CLIP", "DermLIP"])

        if len(all_correct) < SAMPLES_PER_FIGURE and all_pred_correct:
            all_correct.append(sample)
        if len(disagreement) < SAMPLES_PER_FIGURE and all_disagree:
            disagreement.append(sample)
        if (len(minority_fail) < SAMPLES_PER_FIGURE
                and true_cls in MINORITY_CLASSES
                and sup_all_wrong and vlm_all_wrong):
            minority_fail.append(sample)

        if (len(all_correct) >= SAMPLES_PER_FIGURE
                and len(disagreement) >= SAMPLES_PER_FIGURE
                and len(minority_fail) >= SAMPLES_PER_FIGURE):
            break

    return all_correct, disagreement, minority_fail


# ── Full metric evaluation ────────────────────────────────────────────────────
def evaluate_metrics(test_df, sup_models, sup_gradcams,
                     vlm_models, vlm_preprocesses, vlm_txt_feats):
    all_results = {
        name: {"per_class": {cls: {"iou": [], "dice": []} for cls in CLASS_NAMES}}
        for name in MODEL_NAMES
    }
    examples = {name: {} for name in MODEL_NAMES}

    print("\nRunning full metric evaluation...")
    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        img_id = row["image"]
        true_label = int(row[CLASS_NAMES].values.argmax())
        true_cls = CLASS_NAMES[true_label]
        mask = load_mask(img_id)
        if mask is None:
            continue

        img_raw = load_image_raw(img_id)
        img_np = img_to_np(img_raw)
        img_tensor = val_transform(img_raw)

        for name, model in sup_models.items():
            pred = get_sup_pred(model, img_tensor)
            cam = compute_cam_sup(sup_gradcams[name], img_tensor, pred)
            iou, dice = iou_dice(cam, mask)
            all_results[name]["per_class"][true_cls]["iou"].append(iou)
            all_results[name]["per_class"][true_cls]["dice"].append(dice)
            if true_cls not in examples[name]:
                examples[name][true_cls] = {"img_np": img_np, "cam": cam,
                                             "mask": mask, "pred": CLASS_NAMES[pred]}

        for name in ["CLIP", "DermLIP"]:
            pred = get_vlm_pred(vlm_models[name], vlm_preprocesses[name],
                                     vlm_txt_feats[name], img_id)
            cam = compute_cam_vlm(vlm_models[name],
                                        vlm_txt_feats[name],
                                        vlm_preprocesses[name],
                                        img_id, pred)
            iou, dice = iou_dice(cam, mask)
            all_results[name]["per_class"][true_cls]["iou"].append(iou)
            all_results[name]["per_class"][true_cls]["dice"].append(dice)
            if true_cls not in examples[name]:
                examples[name][true_cls] = {"img_np": img_np, "cam": cam,
                                             "mask": mask, "pred": CLASS_NAMES[pred]}

    final = {}
    for name in MODEL_NAMES:
        pc = all_results[name]["per_class"]
        agg_pc, all_iou, all_dice = {}, [], []
        for cls in CLASS_NAMES:
            ious, dices = pc[cls]["iou"], pc[cls]["dice"]
            agg_pc[cls] = {
                "iou":  round(float(np.mean(ious)),  4) if ious  else None,
                "dice": round(float(np.mean(dices)), 4) if dices else None,
            }
            all_iou.extend(ious)
            all_dice.extend(dices)
        final[name] = {
            "per_class": agg_pc,
            "overall": {
                "iou": round(float(np.mean(all_iou)),  4),
                "dice": round(float(np.mean(all_dice)), 4),
            },
        }
    return final, examples


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _, _, test_df = get_splits()

    print("Building all models...")
    (sup_models, sup_gradcams,
     vlm_models, vlm_preprocesses, vlm_txt_feats) = build_all_models()

    # cross-model figures
    all_correct, disagreement, minority_fail = collect_samples(
        test_df, sup_models, sup_gradcams,
        vlm_models, vlm_preprocesses, vlm_txt_feats
    )

    if all_correct:
        make_report_figure(all_correct,
                           "Figure 1 — All Models Localize Correctly",
                           "fig1_all_correct.png", MODEL_NAMES)
    if disagreement:
        make_report_figure(disagreement,
                           "Figure 2 — Model Disagreement Case",
                           "fig2_disagreement.png", MODEL_NAMES)
    if minority_fail:
        make_report_figure(minority_fail,
                           "Figure 3 — All Models Fail on Minority Class (DF / VASC)",
                           "fig3_minority_fail.png", MODEL_NAMES)

    vlm_samples = all_correct[:SAMPLES_PER_FIGURE] or disagreement[:SAMPLES_PER_FIGURE]
    if vlm_samples:
        make_report_figure(vlm_samples,
                           "Figure 4 — CLIP vs DermLIP Attention",
                           "fig4_clip_vs_dermlip.png", ["CLIP", "DermLIP"])

    sup_samples = all_correct[:SAMPLES_PER_FIGURE] or disagreement[:SAMPLES_PER_FIGURE]
    if sup_samples:
        make_report_figure(sup_samples,
                           "Figure 5 — CNN vs ResNet-50 vs Swin-T",
                           "fig5_supervised.png", ["CNN", "ResNet-50", "Swin-T"])

    # full eval + per-model examples
    metrics, examples = evaluate_metrics(
        test_df, sup_models, sup_gradcams,
        vlm_models, vlm_preprocesses, vlm_txt_feats
    )

    with open(OUTPUT_DIR / "gradcam_summary.json", "w") as f:
        json.dump(metrics, f, indent=2)

    for name in MODEL_NAMES:
        model_dir = OUTPUT_DIR / name.lower().replace("-", "")
        model_dir.mkdir(parents=True, exist_ok=True)
        with open(model_dir / "gradcam_metrics.json", "w") as f:
            json.dump(metrics[name], f, indent=2)
        save_per_model_examples(name, examples[name], model_dir)

    print(f"\n{'Model':<12}  {'IoU':>8}  {'Dice':>8}")
    print("-" * 32)
    for name in MODEL_NAMES:
        r = metrics[name]["overall"]
        print(f"{name:<12}  {r['iou']:>8.4f}  {r['dice']:>8.4f}")

    print(f"\nSummary: {OUTPUT_DIR / 'gradcam_summary.json'}")
    print(f"Figures: {OUTPUT_DIR / 'figures'}/")
