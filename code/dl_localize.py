"""
dl_localize.py
==============
DL ablation for Drift-Sense: a small Siamese CNN, trained from scratch on
OUR OWN synthetic generator (no external/pretrained weights -- this matters
because the problem statement explicitly forbids external/proprietary
datasets, and this environment can't reach pretrained-model hosts like
torch.hub or Hugging Face anyway, which is itself a realistic constraint:
a real fab would not want to ship its images to an external model host
either).

WHY A SIAMESE CNN AND NOT A BIGGER PRETRAINED BACKBONE (e.g. DINOv2/ViT)
-------------------------------------------------------------------------
A frozen large pretrained vision backbone (ImageNet/DINOv2-style) is
tempting to reach for as "the latest tech," but it's the wrong tool here:
those backbones are trained on natural-image statistics (edges, textures,
object parts) and have no reason to represent nanoscale periodic
line/contact structure any better than a much smaller, task-specific
network -- and you cannot fine-tune a frozen backbone without significant
compute you likely don't have in a hackathon sandbox. A small CNN trained
directly on the target domain, with a contrastive objective explicitly
designed to distinguish true site from periodic look-alikes, is both more
defensible and cheaper. This ablation exists to produce an honest
speed/accuracy comparison against classical FFT-NCC, not to chase
architecture novelty for its own sake.

ARCHITECTURE
------------
A small fully-convolutional embedding network (4 conv layers, stride-2
downsampling) applied to both:
  - the downsampled reference template (target embedding, single vector
    via global average pool)
  - the full search image (dense feature map, same network, no pooling)
then a dense cosine-similarity map between the target embedding and every
spatial location of the search feature map -- the DL analogue of the
matched-filtering step in localize.py, but with LEARNED features instead
of raw pixel intensities. This is the same family of idea as SiamFC
(fully-convolutional Siamese network tracking), adapted to this
localization task.

TRAINING
--------
Self-supervised-style contrastive training directly on generate_dataset.py
output: for each training pair, the embedded true location is the positive,
and the periodic-background locations elsewhere in the same search image
are the (hard) negatives -- exactly the disambiguation the classical NCC
struggles with in the "hard" difficulty setting.

USAGE
-----
    python dl_localize.py --train --epochs 15 --out siamese.pt
    python dl_localize.py --eval --weights siamese.pt --n 30
"""

import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from generate_dataset import generate_pair
from scipy.ndimage import zoom as ndi_zoom


class EmbeddingNet(nn.Module):
    """Small fully-convolutional embedding network. Stride-2 conv layers
    downsample by 16x total, so a 1000x1000 search image -> ~62x62 feature
    map, and a 100x100 template -> ~6x6 -> global-average-pooled to a
    single embedding vector."""

    def __init__(self, embed_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 5, stride=2, padding=2), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, 5, stride=2, padding=2), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, embed_dim, 3, stride=2, padding=1),
        )

    def forward(self, x):
        return self.net(x)  # B x embed_dim x H' x W' (dense feature map)


def embed_template(net, template01):
    """template01: HxW float32 in [0,1] -> single L2-normalized embedding."""
    t = torch.from_numpy(template01).float()[None, None]  # 1x1xHxW
    feat = net(t)  # 1 x D x h x w
    vec = feat.mean(dim=(2, 3))  # global average pool -> 1 x D
    return F.normalize(vec, dim=1)


def dense_similarity_map(net, search01):
    """search01: HxW float32 in [0,1] -> dense L2-normalized feature map
    (1 x D x h x w)."""
    s = torch.from_numpy(search01).float()[None, None]
    feat = net(s)
    return F.normalize(feat, dim=1)


def localize_dl(net, reference01, search01, zoom_ratio, expected_xy=None,
                 max_drift_px=None, refine=True):
    """DL analogue of localize.localize(): embed the reference, compute a
    dense cosine-similarity map against the search image's feature map,
    upsample back to pixel coordinates, and take the argmax.

    expected_xy / max_drift_px : same "navigation-error recovery" prior-
        window restriction as the classical localizer -- mask out
        similarity-map cells outside the expected drift radius BEFORE
        taking the argmax, so a genuinely identical decoy elsewhere in
        the image can't win just because a from-scratch CNN hasn't fully
        learned to disambiguate it yet.

    refine : if True, treat the DL argmax as a COARSE pick only (its
        native resolution is capped by the network's stride, ~16px here)
        and run a small classical FFT-NCC search in a local window around
        it for sub-pixel precision. This is a genuine hybrid: DL gives
        fast + prior-aware coarse localization, classical NCC gives back
        the precision, and because the classical step only runs on a
        small crop (not the full 1000x1000 image) it stays fast too.
    """
    net.eval()
    t0 = time.time()
    with torch.no_grad():
        template = ndi_zoom(reference01, 1.0 / zoom_ratio, order=1)
        target_vec = embed_template(net, template)             # 1 x D
        search_feat = dense_similarity_map(net, search01)       # 1 x D x h x w

        sim = torch.einsum("d,dhw->hw", target_vec[0], search_feat[0])  # h x w
        sim_np = sim.numpy()

        scale_y = search01.shape[0] / sim_np.shape[0]
        scale_x = search01.shape[1] / sim_np.shape[1]

        if expected_xy is not None and max_drift_px is not None:
            h, w = sim_np.shape
            yy, xx = np.mgrid[0:h, 0:w]
            cell_x = (xx + 0.5) * scale_x
            cell_y = (yy + 0.5) * scale_y
            dist = np.hypot(cell_x - expected_xy[0], cell_y - expected_xy[1])
            mask = dist <= max_drift_px
            if mask.any():
                masked = np.where(mask, sim_np, -np.inf)
                py, px = np.unravel_index(np.argmax(masked), masked.shape)
            else:
                py, px = np.unravel_index(np.argmax(sim_np), sim_np.shape)
        else:
            py, px = np.unravel_index(np.argmax(sim_np), sim_np.shape)

        x = (px + 0.5) * scale_x
        y = (py + 0.5) * scale_y

        if refine:
            # local classical NCC refinement around the DL coarse pick
            from localize import normalized_cross_correlation_fft, subpixel_peak
            half = template.shape[0]  # crop radius ~ one template width
            cy0, cy1 = max(0, int(y - half)), min(search01.shape[0], int(y + half))
            cx0, cx1 = max(0, int(x - half)), min(search01.shape[1], int(x + half))
            crop = search01[cy0:cy1, cx0:cx1]
            if crop.shape[0] > template.shape[0] and crop.shape[1] > template.shape[1]:
                surface = normalized_cross_correlation_fft(crop, template)
                ry, rx = np.unravel_index(np.argmax(surface), surface.shape)
                ry, rx = subpixel_peak(surface, ry, rx)
                x, y = cx0 + rx, cy0 + ry

    elapsed = time.time() - t0
    return {"chosen": {"x": float(x), "y": float(y), "score": float(sim_np.max())},
            "compute_time_s": elapsed}


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------

def sample_training_batch(net_out_stride, batch_size, size=500, zoom_ratio=10, seed=None):
    """Build a batch of (template, search, positive_yx, negative_yx_list)
    using our own generator, in HARD mode so negatives are genuinely
    close look-alikes -- the useful regime to train against."""
    rng = np.random.default_rng(seed)
    batch = []
    for _ in range(batch_size):
        s = int(rng.integers(0, 1_000_000))
        ref, search, gt = generate_pair(size=size, seed=s, zoom_ratio=zoom_ratio,
                                         difficulty="hard")
        template = ndi_zoom(ref, 1.0 / zoom_ratio, order=1)
        batch.append((template, search, gt))
    return batch


def train(epochs=15, batch_size=4, lr=1e-3, size=300, out_path="siamese.pt", seed=0,
          max_seconds=None):
    net = EmbeddingNet()
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    print(f"Training Siamese embedding net: up to {epochs} epochs, "
          f"{sum(p.numel() for p in net.parameters())} parameters, CPU, "
          f"size={size}px{'' if max_seconds is None else f', time cap={max_seconds}s'}\n")

    t_start = time.time()
    epoch = 0
    for epoch in range(1, epochs + 1):
        if max_seconds is not None and (time.time() - t_start) > max_seconds:
            print(f"  [time cap reached at epoch {epoch}]")
            break
        batch = sample_training_batch(None, batch_size, size=size, seed=seed * 10_000 + epoch)
        total_loss = 0.0
        opt.zero_grad()
        for template, search, gt in batch:
            target_vec = embed_template(net, template)          # 1 x D
            search_feat = dense_similarity_map(net, search)[0]  # D x h x w
            D, h, w = search_feat.shape

            sim = torch.einsum("d,dhw->hw", target_vec[0], search_feat)  # h x w

            scale_y = search.shape[0] / h
            scale_x = search.shape[1] / w
            gy, gx = int(gt["y"] / scale_y), int(gt["x"] / scale_x)
            gy, gx = min(max(gy, 0), h - 1), min(max(gx, 0), w - 1)

            # InfoNCE-style contrastive loss: the true grid cell should
            # have the highest similarity among ALL spatial locations
            # (including the genuinely-similar decoy cell elsewhere).
            logits = (sim.flatten() / 0.1)  # temperature-scaled
            target_idx = torch.tensor(gy * w + gx)
            loss = F.cross_entropy(logits[None], target_idx[None])
            total_loss = total_loss + loss

        total_loss = total_loss / batch_size
        total_loss.backward()
        opt.step()

        if epoch % max(1, epochs // 20) == 0 or epoch == 1:
            print(f"  epoch {epoch:3d}/{epochs}  loss={total_loss.item():.4f}  "
                  f"t={time.time()-t_start:.0f}s")

    torch.save(net.state_dict(), out_path)
    print(f"\nSaved trained weights to {out_path} after {epoch} epochs, "
          f"{time.time()-t_start:.0f}s total")
    return net


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--weights", default="siamese.pt")
    ap.add_argument("--out", default="siamese.pt")
    ap.add_argument("--n", type=int, default=30, help="number of eval cases")
    ap.add_argument("--tolerance", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--size", type=int, default=1000, help="canvas size for train/eval; "
                     "MUST match between the two runs (fully-conv net is scale-sensitive "
                     "to absolute template pixel size, not just physical nm/px)")
    ap.add_argument("--max_seconds", type=float, default=None)
    ap.add_argument("--difficulty", choices=["easy", "hard"], default="hard")
    ap.add_argument("--use_prior", action="store_true")
    ap.add_argument("--drift_std_px", type=float, default=15.0)
    ap.add_argument("--max_drift_px", type=float, default=60.0)
    ap.add_argument("--refine", action="store_true", default=True)
    ap.add_argument("--no_refine", dest="refine", action="store_false")
    args = ap.parse_args()

    if args.train:
        train(epochs=args.epochs, out_path=args.out, seed=args.seed, size=args.size,
              max_seconds=args.max_seconds)

    if args.eval:
        net = EmbeddingNet()
        net.load_state_dict(torch.load(args.weights))
        net.eval()

        prior_rng = np.random.default_rng(args.seed + 777)
        successes, errors, times = 0, [], []
        for i in range(1, args.n + 1):
            ref, search, gt = generate_pair(size=args.size, seed=args.seed + 5000 + i,
                                             difficulty=args.difficulty)
            expected_xy = None
            if args.use_prior:
                expected_xy = (gt["x"] + prior_rng.normal(0, args.drift_std_px),
                                gt["y"] + prior_rng.normal(0, args.drift_std_px))
            result = localize_dl(net, ref, search, zoom_ratio=10,
                                  expected_xy=expected_xy,
                                  max_drift_px=args.max_drift_px if args.use_prior else None,
                                  refine=args.refine)
            c = result["chosen"]
            err = float(np.hypot(c["x"] - gt["x"], c["y"] - gt["y"]))
            success = err <= args.tolerance
            successes += success
            errors.append(err)
            times.append(result["compute_time_s"] * 1000)
            print(f"[{i:2d}/{args.n}] err={err:7.2f}px  t={times[-1]:6.1f}ms  "
                  f"{'OK' if success else 'MISS'}")

        print("\n" + "=" * 60)
        print(f"DL (Siamese CNN) on {args.n} {args.difficulty.upper()} cases "
              f"(tolerance={args.tolerance:.1f}px, prior={args.use_prior}, refine={args.refine}):")
        print(f"  success rate     : {100*successes/args.n:.1f}% ({successes}/{args.n})")
        print(f"  mean/median error: {np.mean(errors):.2f}px / {np.median(errors):.2f}px")
        print(f"  mean compute time: {np.mean(times):.1f}ms")


if __name__ == "__main__":
    main()
