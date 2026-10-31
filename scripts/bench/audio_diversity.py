#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Objective diversity metrics for a corpus of 16-bit stereo WAV clips.

This is a local experiment helper, not part of the CI benchmark suite. It uses
only the Python standard library and a small radix-2 FFT so it can run on a
machine without NumPy.

Expected layout:

    <root>/ff_fixed/*.wav
    <root>/ff_varied/*.wav
    <root>/rs/*.wav

For every clip the script computes a 1024-sample Hann-windowed spectrum per
analysis frame (hop 1024, mono downmix), then:

- mean spectral centroid in Hz;
- mean 13-coefficient mel-frequency cepstrum (26 triangular mel filters,
  DCT-II, coefficients 1..13);
- mean log energy in eight bands (0-200-400-800-1600-3200-6400-12000-24000 Hz);
- unique-frame ratio: distinct quantized (MFCC, band-energy) vectors divided by
  the frame count.

Diversity is the mean pairwise Euclidean distance between clip-level feature
vectors, plus the mean per-clip unique-frame ratio and the global unique-frame
ratio across the whole set.
"""

import argparse
import cmath
import math
import struct
import wave
from pathlib import Path

FRAME = 1024
HOP = 1024
BANDS = [0, 200, 400, 800, 1600, 3200, 6400, 12000, 24000]
MEL_FILTERS = 26
MFCC = 13


def hann(size):
    return [0.5 - 0.5 * math.cos(2.0 * math.pi * i / (size - 1)) for i in range(size)]


WINDOW = hann(FRAME)


def fft(values):
    """In-place iterative radix-2 FFT of a power-of-two-length list."""
    n = len(values)
    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            values[i], values[j] = values[j], values[i]
    length = 2
    while length <= n:
        angle = -2.0 * math.pi / length
        step = cmath.exp(1j * angle)
        for start in range(0, n, length):
            twiddle = 1.0 + 0.0j
            half = length // 2
            for offset in range(half):
                even = values[start + offset]
                odd = values[start + offset + half] * twiddle
                values[start + offset] = even + odd
                values[start + offset + half] = even - odd
                twiddle *= step
        length <<= 1


def read_wav(path):
    with wave.open(str(path), "rb") as handle:
        assert handle.getnchannels() == 2, path
        assert handle.getsampwidth() == 2, path
        rate = handle.getframerate()
        raw = handle.readframes(handle.getnframes())
    samples = struct.unpack("<%dh" % (len(raw) // 2), raw)
    mono = [(samples[i] + samples[i + 1]) * 0.5 / 32768.0 for i in range(0, len(samples), 2)]
    return rate, mono


def mel(frequency):
    return 2595.0 * math.log10(1.0 + frequency / 700.0)


def mel_inverse(value):
    return 700.0 * (10.0 ** (value / 2595.0) - 1.0)


def mel_filters(rate):
    low = mel(20.0)
    high = mel(min(rate / 2.0, 20000.0))
    points = [mel_inverse(low + (high - low) * i / (MEL_FILTERS + 1)) for i in range(MEL_FILTERS + 2)]
    bins = [int(point / (rate / FRAME)) for point in points]
    filters = []
    for index in range(MEL_FILTERS):
        left, center, right = bins[index], bins[index + 1], bins[index + 2]
        weights = [0.0] * (FRAME // 2 + 1)
        for k in range(left, center):
            if center > left:
                weights[k] = (k - left) / (center - left)
        for k in range(center, right):
            if right > center:
                weights[k] = (right - k) / (right - center)
        filters.append(weights)
    return filters


def dct(values):
    result = []
    n = len(values)
    for coefficient in range(1, MFCC + 1):
        total = 0.0
        for index, value in enumerate(values):
            total += value * math.cos(math.pi * coefficient * (index + 0.5) / n)
        result.append(total)
    return result


def features(path):
    rate, mono = read_wav(path)
    filters = mel_filters(rate)
    band_bins = [int(edge / (rate / FRAME)) for edge in BANDS]
    centroids = []
    mfccs = []
    bands = []
    frames = []
    for start in range(0, max(1, len(mono) - FRAME + 1), HOP):
        chunk = mono[start:start + FRAME]
        if len(chunk) < FRAME:
            chunk = chunk + [0.0] * (FRAME - len(chunk))
        spectrum = [complex(value * WINDOW[i], 0.0) for i, value in enumerate(chunk)]
        fft(spectrum)
        power = [abs(spectrum[k]) ** 2 for k in range(FRAME // 2 + 1)]
        total = sum(power)
        if total <= 1e-12:
            centroids.append(0.0)
        else:
            centroids.append(sum(k * (rate / FRAME) * power[k] for k in range(len(power))) / total)
        log_filters = []
        for weights in filters:
            energy = sum(power[k] * weights[k] for k in range(len(power)))
            log_filters.append(math.log(energy + 1e-12))
        mfccs.append(dct(log_filters))
        band = []
        for index in range(len(BANDS) - 1):
            energy = sum(power[band_bins[index]:band_bins[index + 1]])
            band.append(10.0 * math.log10(energy + 1e-12))
        bands.append(band)
        key = tuple(round(value, 3) for value in mfccs[-1]) + tuple(round(value, 2) for value in band)
        frames.append(key)
    count = len(mfccs)
    centroid = sum(centroids) / count
    mfcc_mean = [sum(row[i] for row in mfccs) / count for i in range(MFCC)]
    band_mean = [sum(row[i] for row in bands) / count for i in range(len(BANDS) - 1)]
    return {
        "centroid": centroid,
        "mfcc": mfcc_mean,
        "bands": band_mean,
        "unique": len(set(frames)) / count,
        "frames": frames,
    }


def distance(left, right):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


def summarize(name, paths):
    data = [features(path) for path in sorted(paths)]
    pairs = 0
    centroid_total = 0.0
    mfcc_total = 0.0
    band_total = 0.0
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            centroid_total += abs(data[i]["centroid"] - data[j]["centroid"])
            mfcc_total += distance(data[i]["mfcc"], data[j]["mfcc"])
            band_total += distance(data[i]["bands"], data[j]["bands"])
            pairs += 1
    all_frames = [key for entry in data for key in entry["frames"]]
    return {
        "name": name,
        "clips": len(data),
        "pairs": pairs,
        "centroid": centroid_total / pairs,
        "mfcc": mfcc_total / pairs,
        "bands": band_total / pairs,
        "unique_mean": sum(entry["unique"] for entry in data) / len(data),
        "unique_global": len(set(all_frames)) / len(all_frames),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    results = []
    for name in ["ff_fixed", "ff_varied", "rs"]:
        results.append(summarize(name, (args.root / name).glob("*.wav")))
    print(f"{'set':<12}{'clips':>6}{'pairs':>7}{'centroid Hz':>14}{'MFCC dist':>12}{'band dB':>10}{'unique/clip':>13}{'unique/global':>15}")
    for row in results:
        print(
            f"{row['name']:<12}{row['clips']:>6}{row['pairs']:>7}"
            f"{row['centroid']:>14.1f}{row['mfcc']:>12.2f}{row['bands']:>10.2f}"
            f"{row['unique_mean']:>13.3f}{row['unique_global']:>15.3f}"
        )


if __name__ == "__main__":
    main()
