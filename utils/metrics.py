"""
metrics.py
──────────
Evaluation metrics for both:
    - Rating prediction  (MAE, RMSE)
    - Image generation   (FID, SSIM, LPIPS — lightweight wrappers)
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error


def compute_rating_metrics(r_hat: np.ndarray, r_true: np.ndarray) -> dict[str, float]:
    """Returns MAE and RMSE for rating prediction."""
    mae  = mean_absolute_error(r_true, r_hat)
    rmse = float(np.sqrt(np.mean((r_hat - r_true) ** 2)))
    return {"mae": mae, "rmse": rmse}


def compute_ssim(img1, img2) -> float:
    """Structural Similarity Index between two PIL images."""
    try:
        from skimage.metrics import structural_similarity as ssim
        import numpy as np
        a = np.array(img1.convert("L"))
        b = np.array(img2.convert("L"))
        score, _ = ssim(a, b, full=True)
        return float(score)
    except ImportError:
        return -1.0   # skimage not installed


def compute_psnr(img1, img2) -> float:
    """Peak Signal-to-Noise Ratio between two PIL images."""
    import numpy as np
    a = np.array(img1).astype(np.float32)
    b = np.array(img2).astype(np.float32)
    mse = np.mean((a - b) ** 2)
    if mse == 0:
        return float("inf")
    return float(20 * np.log10(255.0 / np.sqrt(mse)))


def compute_fid_score(real_images, fake_images, device="cuda") -> float:
    """
    Fréchet Inception Distance.
    Requires pytorch-fid  (pip install pytorch-fid).
    real_images / fake_images: list of PIL.Image
    """
    try:
        import torch
        from torchvision import transforms
        from pytorch_fid.fid_score import calculate_frechet_distance
        from pytorch_fid.inception import InceptionV3

        transform = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
        ])

        def get_activations(imgs):
            model = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(device)
            model.eval()
            tensors = torch.stack([transform(img) for img in imgs]).to(device)
            with torch.no_grad():
                acts = model(tensors)[0]
            return acts.squeeze(3).squeeze(2).cpu().numpy()

        real_acts = get_activations(real_images)
        fake_acts = get_activations(fake_images)

        mu_r, sig_r = real_acts.mean(0), np.cov(real_acts, rowvar=False)
        mu_f, sig_f = fake_acts.mean(0), np.cov(fake_acts, rowvar=False)
        return float(calculate_frechet_distance(mu_r, sig_r, mu_f, sig_f))
    except ImportError:
        return -1.0
