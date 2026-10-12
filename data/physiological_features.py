"""
Component B (physiologically-grounded auxiliary pretext task): a small set
of classical, fully UNSUPERVISED signal descriptors computed directly from
the raw waveform via signal processing (no labels, no learned parameters).

These are intentionally kept UNITLESS / sampling-rate-agnostic (fractions
of Nyquist, normalized entropies, etc.) rather than converted to physical
units like BPM, because this codebase spans datasets whose true sampling
rate is not uniformly known or comparable (UCR/UEA archive problems,
MIT-BIH beats, PTB-XL). They are meant to capture "shape of the rhythm and
waveform" in a way that is complementary to, and independent of, whatever
the contrastive encoder itself learns -- giving the model an auxiliary,
domain-motivated training signal (Section: "Physiologically-grounded
auxiliary pretext task" in the redesign) and a second, non-learned
similarity space for validating that retrieved diversity is physiologically
meaningful, not just an embedding-space artifact.

Descriptors (6, each roughly in a bounded range):
  1. dominant_freq_frac : location of the dominant spectral peak, as a
     fraction of the Nyquist frequency -- proxy for "how fast" the
     rhythm is, independent of absolute sampling rate.
  2. spectral_entropy    : Shannon entropy of the normalised power
     spectrum -- low for a clean periodic rhythm, high for noisy/complex
     signals.
  3. sample_entropy      : classic SampEn(m=2, r=0.2*std) nonlinear
     complexity measure, widely used in HRV analysis; computed on a
     decimated copy of the signal to bound the O(n^2) cost.
  4. autocorr_periodicity: strength of the strongest non-trivial
     autocorrelation peak relative to the zero-lag value -- how
     "regular"/periodic the waveform is.
  5. zero_crossing_rate  : fraction of sign changes -- a cheap proxy for
     dominant frequency content / noisiness.
  6. kurtosis            : peakedness of the amplitude distribution --
     distinguishes sharp QRS-like spikes from smooth, rounded waveforms
     (meaningful precisely because signals are already z-normalised
     upstream, so raw energy/RMS is not informative but SHAPE is).
"""
import numpy as np


def _dominant_freq_frac(x: np.ndarray) -> float:
    spec = np.abs(np.fft.rfft(x))
    spec[0] = 0.0  # drop DC
    if spec.sum() <= 0:
        return 0.0
    peak_bin = int(np.argmax(spec))
    return peak_bin / max(1, len(spec) - 1)


def _spectral_entropy(x: np.ndarray) -> float:
    spec = np.abs(np.fft.rfft(x)) ** 2
    spec = spec[1:]  # drop DC
    total = spec.sum()
    if total <= 0:
        return 0.0
    p = spec / total
    p = p[p > 0]
    entropy = -np.sum(p * np.log(p))
    return float(entropy / np.log(len(p)) if len(p) > 1 else 0.0)


def _sample_entropy(x: np.ndarray, m: int = 2, r_frac: float = 0.2) -> float:
    n = len(x)
    if n < m + 2:
        return 0.0
    r = r_frac * (np.std(x) + 1e-8)

    def _count_matches(m_):
        templates = np.array([x[i:i + m_] for i in range(n - m_)])
        count = 0
        for i in range(len(templates)):
            dists = np.max(np.abs(templates[i + 1:] - templates[i]), axis=1) if i + 1 < len(templates) else np.array([])
            count += np.sum(dists <= r)
        return count

    B = _count_matches(m)
    A = _count_matches(m + 1)
    if B == 0 or A == 0:
        return 0.0
    return float(-np.log(A / B))


def _autocorr_periodicity(x: np.ndarray) -> float:
    n = len(x)
    x = x - x.mean()
    ac = np.correlate(x, x, mode="full")[n - 1:]
    if ac[0] <= 0:
        return 0.0
    ac = ac / ac[0]
    lo, hi = max(2, int(0.03 * n)), max(3, int(0.5 * n))
    if hi <= lo:
        return 0.0
    return float(np.clip(np.max(ac[lo:hi]), 0.0, 1.0))


def _zero_crossing_rate(x: np.ndarray) -> float:
    signs = np.sign(x)
    signs[signs == 0] = 1
    crossings = np.sum(np.abs(np.diff(signs)) > 0)
    return float(crossings / max(1, len(x) - 1))


def _kurtosis(x: np.ndarray) -> float:
    std = np.std(x) + 1e-8
    return float(np.mean(((x - x.mean()) / std) ** 4) - 3.0)


def extract_physio_features(x: np.ndarray, max_len: int = 250) -> np.ndarray:
    """x: 1-D float array (already z-normalised, as this codebase does at
    load time). Returns a fixed-size float32 array of 6 descriptors."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if len(x) > max_len:
        idx = np.linspace(0, len(x) - 1, max_len).astype(int)
        x_decim = x[idx]
    else:
        x_decim = x

    feats = np.array([
        _dominant_freq_frac(x),
        _spectral_entropy(x),
        _sample_entropy(x_decim),
        _autocorr_periodicity(x),
        _zero_crossing_rate(x),
        _kurtosis(x),
    ], dtype=np.float32)
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    return feats


N_PHYSIO_FEATURES = 6
