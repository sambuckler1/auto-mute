"""Score audio windows with YAMNet and turn the scores into a stable game/ad decision."""

import csv
import urllib.request
from collections import deque
from math import gcd
from pathlib import Path

import numpy as np
from ai_edge_litert.interpreter import Interpreter
from scipy.signal import resample_poly

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/audio_classifier/yamnet/float32/latest/yamnet.tflite"
CLASS_MAP_URL = "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv"

YAMNET_RATE = 16000
YAMNET_WINDOW = 15600

# Commentary drowns out the crowd most of the time, so music (which nearly every ad has
# and live play rarely does) is the main ad signal; game sounds only count against it.
MUSIC_CLASSES = ("Music", "Background music", "Theme music", "Jingle (music)")
GAME_CLASSES = ("Crowd", "Cheering", "Applause", "Hubbub, speech noise, speech babble",
                "Whistle", "Basketball bounce")


def ensure_model(model_dir: Path) -> tuple[Path, Path]:
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "yamnet.tflite"
    class_map_path = model_dir / "yamnet_class_map.csv"
    for url, path in ((MODEL_URL, model_path), (CLASS_MAP_URL, class_map_path)):
        if not path.exists():
            print(f"Downloading {path.name}...")
            urllib.request.urlretrieve(url, path)
    return model_path, class_map_path


class YamnetDetector:
    def __init__(self, model_path: Path, class_map_path: Path):
        self._interpreter = Interpreter(model_path=str(model_path), num_threads=2)
        self._interpreter.allocate_tensors()
        self._input = self._interpreter.get_input_details()[0]["index"]
        self._output = self._interpreter.get_output_details()[0]["index"]
        with open(class_map_path, newline="") as f:
            self.labels = [row["display_name"].strip() for row in csv.DictReader(f)]
        self._game = [self.labels.index(name) for name in GAME_CLASSES]
        self._music = [self.labels.index(name) for name in MUSIC_CLASSES]
        self._speech = self.labels.index("Speech")

    def analyze(self, waveform: np.ndarray) -> dict:
        """Score one YAMNet-sized window of 16 kHz mono audio."""
        self._interpreter.set_tensor(self._input, waveform.astype(np.float32))
        self._interpreter.invoke()
        scores = self._interpreter.get_tensor(self._output)[0]
        game = float(min(1.0, scores[self._game].sum()))
        music = float(scores[self._music].max())
        return {
            "ad": max(0.0, music - game),
            "music": music,
            "game": game,
            "speech": float(scores[self._speech]),
            "top": self.labels[int(scores.argmax())],
            "scores": scores,
        }


class Windower:
    """Buffers incoming audio and emits a resampled YAMNet window every hop."""

    def __init__(self, input_rate: int, hop_s: float = 0.5):
        divisor = gcd(YAMNET_RATE, input_rate)
        self._up, self._down = YAMNET_RATE // divisor, input_rate // divisor
        self._window_len = int(np.ceil(YAMNET_WINDOW * input_rate / YAMNET_RATE))
        self._hop_len = int(hop_s * input_rate)
        self._buffer = np.zeros(0, dtype=np.float32)
        self._since_hop = 0

    def push(self, block: np.ndarray) -> np.ndarray | None:
        self._buffer = np.concatenate((self._buffer, block))[-self._window_len:]
        self._since_hop += len(block)
        if self._since_hop < self._hop_len or len(self._buffer) < self._window_len:
            return None
        self._since_hop = 0
        return resample_poly(self._buffer, self._up, self._down)[:YAMNET_WINDOW].astype(np.float32)


class AdDecider:
    """Smooths the per-window ad score and flips state only after it stays across a threshold.

    Separate thresholds for entering and leaving an ad, plus minimum durations, keep the
    mute from flickering on a short music sting in the game or a quiet moment in an ad.
    """

    def __init__(self, ad_above: float, game_below: float, ad_after_s: float, game_after_s: float,
                 smooth_s: float, hop_s: float):
        self.ad_above, self.game_below = ad_above, game_below
        self.ad_after_s, self.game_after_s = ad_after_s, game_after_s
        self._history: deque[float] = deque(maxlen=max(1, round(smooth_s / hop_s)))
        self._pending_since: float | None = None
        self.is_ad = False
        self.smoothed = 0.0

    def update(self, ad_score: float, now: float) -> bool:
        self._history.append(ad_score)
        self.smoothed = sum(self._history) / len(self._history)

        if self.is_ad:
            wants_flip, wait = self.smoothed < self.game_below, self.game_after_s
        else:
            wants_flip, wait = self.smoothed > self.ad_above, self.ad_after_s

        if not wants_flip:
            self._pending_since = None
        elif self._pending_since is None:
            self._pending_since = now
        elif now - self._pending_since >= wait:
            self.is_ad = not self.is_ad
            self._pending_since = None
        return self.is_ad

    def hold(self) -> None:
        """Keep the current state, e.g. while the audio is silent."""
        self._pending_since = None
