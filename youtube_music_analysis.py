#!/usr/bin/env python3
"""Analyze a segment of a YouTube video for musical features.

This script downloads a YouTube video's audio, extracts a specified
segment, and estimates simple musical information:
- BPM (tempo)
- Key
- Number of distinct songs within the segment
- Whether any songs repeat

Dependencies: yt-dlp, ffmpeg, librosa, numpy

Example:
    python youtube_music_analysis.py --url <VIDEO_URL> --start 00:01:00 --end 00:02:30
"""

import argparse
import os
import subprocess
import tempfile
from typing import List

import librosa
import numpy as np


def download_audio(url: str, out_path: str) -> str:
    """Download audio from a YouTube video as WAV using yt-dlp."""
    # yt-dlp replaces %(ext)s with file extension
    template = out_path + ".%(ext)s"
    cmd = ["yt-dlp", "-x", "--audio-format", "wav", "-o", template, url]
    subprocess.run(cmd, check=True)
    return out_path + ".wav"


def clip_audio(src: str, start: float, end: float, dst: str) -> None:
    """Use ffmpeg to cut src between start and end seconds to dst."""
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        src,
        "-ss",
        str(start),
        "-to",
        str(end),
        dst,
    ]
    subprocess.run(cmd, check=True)


def fingerprint_segments(y: np.ndarray, sr: int, segment_len: float = 30.0) -> List[np.ndarray]:
    """Create a simple chroma-based fingerprint per segment."""
    seg_samples = int(segment_len * sr)
    fps: List[np.ndarray] = []
    for i in range(0, len(y), seg_samples):
        segment = y[i : i + seg_samples]
        if len(segment) < seg_samples * 0.5:
            break
        chroma = librosa.feature.chroma_cqt(y=segment, sr=sr)
        fp = chroma.mean(axis=1)
        fp = fp / np.linalg.norm(fp)
        fps.append(fp)
    return fps


def cluster_fingerprints(fps: List[np.ndarray], threshold: float = 0.9) -> List[List[np.ndarray]]:
    """Group fingerprints by correlation similarity."""
    clusters: List[List[np.ndarray]] = []
    for fp in fps:
        placed = False
        for cluster in clusters:
            if any(np.corrcoef(fp, other)[0, 1] >= threshold for other in cluster):
                cluster.append(fp)
                placed = True
                break
        if not placed:
            clusters.append([fp])
    return clusters


def estimate_key(chroma_mean: np.ndarray) -> str:
    keys = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    return keys[int(chroma_mean.argmax())]


def analyze_clip(path: str) -> dict:
    y, sr = librosa.load(path)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    key = estimate_key(chroma.mean(axis=1))
    fps = fingerprint_segments(y, sr)
    clusters = cluster_fingerprints(fps)
    return {
        "bpm": float(tempo),
        "key": key,
        "clusters": clusters,
    }


def parse_time(val: str) -> float:
    """Parse time in seconds or HH:MM:SS format."""
    if ":" in val:
        parts = [float(p) for p in val.split(":")]
        seconds = 0.0
        for part in parts:
            seconds = seconds * 60 + part
        return seconds
    return float(val)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze YouTube audio segments")
    parser.add_argument("--url", required=True, help="YouTube video URL")
    parser.add_argument("--start", required=True, help="Start time (seconds or HH:MM:SS)")
    parser.add_argument("--end", required=True, help="End time (seconds or HH:MM:SS)")
    args = parser.parse_args()

    start = parse_time(args.start)
    end = parse_time(args.end)

    with tempfile.TemporaryDirectory() as tmp:
        raw = os.path.join(tmp, "audio")
        src = download_audio(args.url, raw)
        clip = os.path.join(tmp, "clip.wav")
        clip_audio(src, start, end, clip)
        results = analyze_clip(clip)

    clusters = results["clusters"]
    print(f"Estimated BPM: {results['bpm']:.1f}")
    print(f"Estimated Key: {results['key']}")
    print(f"Estimated number of songs: {len(clusters)}")
    for idx, cluster in enumerate(clusters, 1):
        if len(cluster) > 1:
            print(f"Song {idx} repeats {len(cluster)} times")


if __name__ == "__main__":
    main()
