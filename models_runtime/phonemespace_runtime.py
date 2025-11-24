# models_runtime/phonemespace_runtime.py
import pickle
import numpy as np

class PhonemeSpace:
    def __init__(self, proto_path: str, custom_radii: dict = None):
        """
        Args:
            proto_path: Path to prototype pickle file
            custom_radii: Optional dict of {phone: radius} to override PKL radii
        """
        with open(proto_path, "rb") as f:
            self.protos = pickle.load(f)  # {ph: {"centroid": np.array, "radius": float, ...}}

        self.phones = sorted(self.protos.keys())
        self.centroids = np.stack(
            [self.protos[p]["centroid"] for p in self.phones],
            axis=0
        )  # (P, D)
        
        # Apply custom radii if provided (NEW)
        if custom_radii:
            for phone, radius in custom_radii.items():
                if phone in self.protos:
                    self.protos[phone]["radius"] = radius

    def _l2(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a - b))

    def nearest_phoneme(self, z: np.ndarray):
        """
        Nearest centroid over ALL phones.
        Returns (phone, distance, index).
        """
        z = np.asarray(z, dtype=np.float32)
        diffs = self.centroids - z[None, :]
        dists = np.linalg.norm(diffs, axis=1)  # (P,)
        idx = int(np.argmin(dists))
        return self.phones[idx], float(dists[idx]), idx

    def nearest_other_phoneme(self, z: np.ndarray, target_phone: str):
        """
        Nearest centroid among phones != target_phone.
        Returns (phone, distance) or (None, None) if not applicable.
        """
        z = np.asarray(z, dtype=np.float32)

        diffs = self.centroids - z[None, :]
        dists = np.linalg.norm(diffs, axis=1)  # (P,)

        # mask out target index
        if target_phone in self.phones:
            t_idx = self.phones.index(target_phone)
            dists[t_idx] = np.inf

        idx = int(np.argmin(dists))
        if not np.isfinite(dists[idx]):
            return None, None

        return self.phones[idx], float(dists[idx])

    def classify_embedding(self, z: np.ndarray, target_phone: str | None = None,
                           margin_min: float = 0.3):
        """
        Classify a single phoneme embedding with radius + margin.

        Returns a dict with:
          - nearest_phoneme, dist_to_nearest, radius_nearest, within_nearest_radius
          - target, known_target, dist_to_target, radius_target
          - nearest_other_phoneme, dist_to_nearest_other, margin
          - correct (bool or None)
          - confidence (float in [0,1], heuristic)
        """
        z = np.asarray(z, dtype=np.float32)

        # --- nearest overall ---
        near_p, d_near, _ = self.nearest_phoneme(z)
        r_near = self.protos[near_p]["radius"]

        result = {
            "nearest_phoneme": near_p,
            "dist_to_nearest": d_near,
            "radius_nearest": r_near,
            "within_nearest_radius": d_near <= r_near,
        }

        # If we don't know the target, we just return nearest info + no correctness
        if target_phone is None or target_phone not in self.protos:
            result.update({
                "target": target_phone,
                "known_target": target_phone in self.protos if target_phone is not None else False,
                "dist_to_target": None,
                "radius_target": None,
                "nearest_other_phoneme": None,
                "dist_to_nearest_other": None,
                "margin": None,
                "correct": None,
                "confidence": None,
            })
            return result

        # --- distance to target ---
        mu_t = self.protos[target_phone]["centroid"]
        r_t  = self.protos[target_phone]["radius"]
        d_t  = self._l2(z, mu_t)
        inside = d_t <= r_t

        # --- nearest OTHER phone (for margin) ---
        other_p, d_other = self.nearest_other_phoneme(z, target_phone)
        if other_p is None:
            margin = None
        else:
            # positive margin = target closer than any other
            margin = d_other - d_t

        # --- correctness rule: inside radius AND margin not too negative ---
        if margin is None:
            correct = inside
        else:
            correct = bool(inside and (margin >= -0.1))  # allow a bit of overlap

        # --- confidence (very heuristic) ---
        if margin is None:
            confidence = 0.5 if inside else 0.0
        else:
            # map margin and radial position into [0,1]
            m_clipped = max(-0.5, min(0.5, margin))
            m_norm = (m_clipped + 0.5)  # 0..1
            # closer to center → higher
            radial = max(0.0, min(1.0, 1.0 - d_t / (r_t + 1e-6)))
            confidence = float(0.5 * m_norm + 0.5 * radial)

        result.update({
            "target": target_phone,
            "known_target": True,
            "dist_to_target": d_t,
            "radius_target": r_t,
            "nearest_other_phoneme": other_p,
            "dist_to_nearest_other": d_other,
            "margin": margin,
            "correct": correct,
            "confidence": confidence,
        })

        return result