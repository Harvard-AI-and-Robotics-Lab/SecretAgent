"""Seeded noise fields used by SecretAgent-Tool. No model calls or optimization."""
import math
import numpy as np
from scipy import ndimage, signal
from scipy.spatial import cKDTree
from .distributions import DISTRIBUTION_DOMAINS, distribution_catalog, sample_distribution
from ..schemas import NumericConfig

DOMAINS = {
    "perlin": {"period": (2, 512, 32, False), "octaves": (1, 6, 3, True), "sine_frequency": (0.25, 32, 4, False)},
    "gabor": {"wavelength": (2, 256, 12, False), "sigma": (1, 64, 8, False), "orientation": (0, 180, 45, False), "density": (0.0001, 0.2, 0.02, False)},
    "single_fourier": {"frequency_x": (0, 512, 24, True), "frequency_y": (0, 512, 16, True), "phase": (0, 360, 0, False)},
    "turing": {"kernel_n": (3, 31, 13, True), "kernel_l1": (1, 29, 5, True), "kernel_l2": (1, 29, 5, True), "iterations": (1, 40, 20, True)},
    "worley": {"cell_size": (4, 256, 32, False), "distance_order": (1, 3, 1, True)},
    "wavelet": {"scale": (1, 64, 4, False), "bands": (1, 5, 2, True), "persistence": (0.1, 1, 0.5, False)},
    "phasor": {"wavelength": (2, 256, 12, False), "sigma": (1, 64, 8, False), "orientation": (0, 180, 45, False), "density": (0.0001, 0.2, 0.02, False)},
    "gaussian": {"std": (0.05, 2, 0.5, False)},
    "poisson": {"rate": (1, 10000, 60, False)},
    "speckle": {"std": (0.05, 2, 0.5, False)},
    **DISTRIBUTION_DOMAINS,
}

def catalog():
    # Expose coupled constraints already enforced by parameters()/generate(), not
    # just independent min/max ranges. No changes to numerical generators.
    coupled = {
        "perlin": ["period / 2**(octaves - 1) >= 1 native pixel; e.g. octaves=6 requires period>=32"],
        "turing": ["kernel_l1 < kernel_n and kernel_l2 < kernel_n"],
        "single_fourier": ["frequency_x <= floor(image_width/2)", "frequency_y <= floor(image_height/2)",
                           "frequency_x and frequency_y must not both be zero"],
    }
    image_fields = {
        "gaussian": {
            "field_formula": "Normal(mean=0, std=std), independently per pixel and RGB channel",
            "brightness_dependence": "none",
        },
        "poisson": {
            "field_formula": "(Poisson(x * rate) - x * rate) / sqrt(rate), where x is original intensity in [0,1]",
            "brightness_dependence": "conditional mean 0 and variance x before gains; rate controls discreteness",
        },
        "speckle": {
            "field_formula": "x * Normal(mean=0, std=std), where x is original intensity in [0,1]",
            "brightness_dependence": "conditional mean 0 and standard deviation x * std before gains",
        },
    }
    return {name: {"parameters": {key: {"min": lo, "max": hi, "default": d, "integer": integer}
                                  for key, (lo, hi, d, integer) in params.items()},
                   "coupled_constraints": coupled.get(name, []),
                   "units": "spatial lengths: native pixels; Fourier indices: cycles per image; orientation/phase: degrees",
                   **image_fields.get(name, {}),
                   **(distribution_catalog(name) if name in DISTRIBUTION_DOMAINS else {})}
            for name, params in DOMAINS.items()}

def default_config(noise_type, seed=0):
    return NumericConfig(strength=0.8, seed=seed, feather_pixels=4,
                         color=[1, 1, 1], parameters=[{"name": k, "value": v[2]} for k, v in DOMAINS[noise_type].items()])

def parameters(noise_type, config):
    p = {x.name: x.value for x in config.parameters}
    if set(p) != set(DOMAINS[noise_type]):
        raise ValueError(f"{noise_type}: exactly these parameters required: {list(DOMAINS[noise_type])}")
    for key, (lo, hi, _, integer) in DOMAINS[noise_type].items():
        if not math.isfinite(p[key]) or not lo <= p[key] <= hi or (integer and p[key] != int(p[key])):
            raise ValueError(f"{noise_type}.{key}: outside allowed domain")
        if integer:
            p[key] = int(p[key])
    if noise_type == "turing" and max(p["kernel_l1"], p["kernel_l2"]) >= p["kernel_n"]:
        raise ValueError("Turing L1,L2 must be smaller than N")
    if noise_type == "perlin" and p["period"] / 2**(p["octaves"] - 1) < 1:
        raise ValueError("Perlin smallest octave period must be at least one pixel")
    if noise_type == "single_fourier" and p["frequency_x"] == p["frequency_y"] == 0:
        raise ValueError("Single Fourier requires a non-DC frequency")
    return p

def _signed_range(field):
    lo, hi = field.min(), field.max()
    return np.zeros_like(field) if hi - lo < 1e-12 else 2 * (field - lo) / (hi - lo) - 1

def _perlin(h, w, p, rng):
    total = np.zeros((h, w), np.float32)
    for octave in range(p["octaves"]):
        period = p["period"] / 2**octave
        x = np.arange(w, dtype=np.float32)[None, :] / period
        all_y = np.arange(h, dtype=np.float32)[:, None] / period
        ix, all_iy = np.floor(x).astype(int), np.floor(all_y).astype(int)
        angles = rng.uniform(0, 2 * np.pi, (all_iy.max() + 2, ix.max() + 2))
        gx, gy = np.cos(angles), np.sin(angles)
        # Row blocking changes allocation only, not coordinates, gradient RNG or arithmetic.
        for start in range(0, h, 128):
            stop = min(h, start + 128)
            iy = all_iy[start:stop]
            fx, fy = x - ix, all_y[start:stop] - iy
            n00 = gx[iy, ix] * fx + gy[iy, ix] * fy
            n10 = gx[iy, ix + 1] * (fx - 1) + gy[iy, ix + 1] * fy
            n01 = gx[iy + 1, ix] * fx + gy[iy + 1, ix] * (fy - 1)
            n11 = gx[iy + 1, ix + 1] * (fx - 1) + gy[iy + 1, ix + 1] * (fy - 1)
            u, v = fx**3 * (fx * (fx * 6 - 15) + 10), fy**3 * (fy * (fy * 6 - 15) + 10)
            total[start:stop] += ((1 - v) * ((1 - u) * n00 + u * n10) + v * ((1 - u) * n01 + u * n11)) / 2**octave
    base = (_signed_range(total) + 1) / 2
    return _signed_range(np.sin(base * p["sine_frequency"] * np.pi))

def _sparse_convolution(h, w, p, rng, *, complex_phase):
    radius = int(math.ceil(3 * p["sigma"]))
    yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    angle = math.radians(p["orientation"])
    phase = 2 * np.pi * (xx * math.cos(angle) + yy * math.sin(angle)) / p["wavelength"]
    envelope = np.exp(-(xx**2 + yy**2) / (2 * p["sigma"]**2))
    # Padded independent impulses retain boundary statistics; same seed is reproducible.
    shape = (h + 2 * radius, w + 2 * radius)
    occupied = rng.random(shape) < p["density"]
    weights = rng.uniform(-1, 1, shape)
    if complex_phase:
        impulses = occupied * weights * np.exp(1j * rng.uniform(0, 2 * np.pi, shape))
        field = signal.fftconvolve(impulses, envelope * np.exp(1j * phase), mode="valid")
        return np.sin(np.angle(field))  # Phasor phase, NOT real-valued Gabor noise.
    field = signal.fftconvolve(occupied * weights, envelope * np.cos(phase), mode="valid")
    return _signed_range(field)

def turing_kernel(n, l1, l2):
    kernel = np.zeros((n, n))
    ya, yb, xa, xb = (n - l1) // 2, (n + l1) // 2, (n - l2) // 2, (n + l2) // 2
    kernel[ya:yb, :xa] = kernel[ya:yb, xb:] = -1
    kernel[:ya, xa:xb] = kernel[yb:, xa:xb] = -1
    kernel[ya:yb, xa:xb] = (l1 + l2) * n / l1 / l2 - 2
    return kernel

def _turing(h, w, p, rng):
    state = rng.choice([-1., 1.], (h, w, 3))
    n, l1, l2 = p["kernel_n"], p["kernel_l1"], p["kernel_l2"]
    def box(a, length, axis):
        start = (n - length) // 2
        origin = (n - 1) // 2 - start - length // 2
        return ndimage.uniform_filter1d(a, length, axis=axis, mode="constant", origin=origin) * length
    for _ in range(p["iterations"]):
        mixed = state + np.roll(state, -1, axis=2)
        narrow_x = box(mixed, l2, 1)
        center = box(narrow_x, l1, 0)
        horizontal = box(box(mixed, n, 1), l1, 0)
        vertical = box(narrow_x, n, 0)
        # Exact same cross-shaped integer-scaled correlation, using running sums.
        # Rounding removes division error in box averages and preserves exact zero ties.
        correlation = (l1 + l2) * n * center - l1 * l2 * (horizontal + vertical)
        state = (2 * (np.rint(correlation) > 0) - 1).astype(float)
    return state

def _worley(h, w, p, rng):
    count = max(p["distance_order"], int(round(h * w / p["cell_size"]**2)))
    points = rng.uniform([0, 0], [h, w], (count, 2))
    tree = cKDTree(points, boxsize=[h, w])
    yy, xx = np.mgrid[:h, :w]
    distances = tree.query(np.stack([yy.ravel(), xx.ravel()], axis=1), k=p["distance_order"])[0]
    if p["distance_order"] > 1:
        distances = distances[:, -1]
    return _signed_range(distances.reshape(h, w))

def _wavelet(h, w, p, rng):
    result = np.zeros((h, w))
    norm = 0.
    for band in range(p["bands"]):
        tile = rng.normal(size=(64, 64))
        detail = tile - _coarsen_axis(_coarsen_axis(tile, 0), 1)
        detail = detail + np.roll(detail, (33, 33), axis=(0, 1))
        # order=2, prefilter=False evaluates the SAME quadratic B-spline weights
        # directly on coefficients. C implementation avoids many full-resolution
        # temporary arrays. grid-wrap is the Appendix's periodic modulo rule.
        field = ndimage.affine_transform(detail, np.eye(2) * (2**band / p["scale"]),
                                         output_shape=(h, w), order=2,
                                         mode="grid-wrap", prefilter=False)
        weight = p["persistence"]**band
        result += weight * field
        norm += weight
    return _signed_range(result / norm)

def generate(noise_type, original, config):
    p = parameters(noise_type, config)
    rng = np.random.default_rng(config.seed)
    h, w = original.shape[:2]
    if noise_type == "perlin":
        field = _perlin(h, w, p, rng)
    elif noise_type == "gabor":
        field = _sparse_convolution(h, w, p, rng, complex_phase=False)
    elif noise_type == "phasor":
        field = _sparse_convolution(h, w, p, rng, complex_phase=True)
    elif noise_type == "single_fourier":
        if p["frequency_x"] > w // 2 or p["frequency_y"] > h // 2:
            raise ValueError("Fourier frequencies must not exceed image Nyquist indices")
        y, x = np.mgrid[:h, :w]
        field = np.cos(2 * np.pi * (p["frequency_x"] * x / w + p["frequency_y"] * y / h) + math.radians(p["phase"]))
    elif noise_type == "turing":
        field = _turing(h, w, p, rng)
    elif noise_type == "worley":
        field = _worley(h, w, p, rng)
    elif noise_type == "wavelet":
        field = _wavelet(h, w, p, rng)
    elif noise_type == "gaussian":
        field = rng.normal(0, p["std"], original.shape)
    elif noise_type == "poisson":
        field = (rng.poisson(original * p["rate"]) - original * p["rate"]) / math.sqrt(p["rate"])
    elif noise_type == "speckle":
        field = original * rng.normal(0, p["std"], original.shape)
    elif noise_type in DISTRIBUTION_DOMAINS:
        field = sample_distribution(noise_type, p, rng, original.shape)
    else:
        raise ValueError(f"Unknown noise: {noise_type}")
    if field.ndim == 2:
        field = np.repeat(field[..., None], 3, axis=2)
    if not np.isfinite(field).all():
        raise ValueError("Non-finite noise field")
    return field.astype(np.float32) * np.asarray(config.color, dtype=np.float32)


WAVELET_COEFFICIENTS = np.array([
    .000334, -.001528, .000410, .003545, -.000938, -.008233, .002172, .019120,
    -.005040, -.044412, .011655, .103311, -.025936, -.243780, .033979, .655340,
    .655340, .033979, -.243780, -.025936, .103311, .011655, -.044412, -.005040,
    .019120, .002172, -.008233, -.000938, .003546, .000410, -.001528, .000334])

def _coarsen_axis(a, axis):
    # Cook/DeRose Appendix 1; use 32 defined taps (-16..15), fixing the printed
    # C loop's off-by-one access to a nonexistent 33rd coefficient.
    n = a.shape[axis]
    low = sum(c * np.take(a, (2 * np.arange(n // 2) + k - 16) % n, axis=axis)
              for k, c in enumerate(WAVELET_COEFFICIENTS))
    i = np.arange(n)
    coefficients = np.array([.25, .75, .75, .25])
    out = np.zeros_like(a)
    shape = [1] * a.ndim
    shape[axis] = n
    for shift in (0, 1):
        k = i // 2 + shift
        weights = coefficients[i - 2 * k + 2]
        out += np.take(low, k % (n // 2), axis=axis) * weights.reshape(shape)
    return out
