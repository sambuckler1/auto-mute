"""Route system audio from BlackHole to the real output device, with a mute control."""

import queue
import subprocess
import threading
from contextlib import contextmanager
from pathlib import Path

import AVFoundation
import numpy as np
import sounddevice as sd

BLACKHOLE = "BlackHole 2ch"

MIC_NOT_DETERMINED, MIC_RESTRICTED, MIC_DENIED, MIC_AUTHORIZED = range(4)


def ensure_microphone_access() -> None:
    """macOS feeds silence to inputs (including BlackHole) without Microphone permission."""
    device = AVFoundation.AVCaptureDevice
    status = device.authorizationStatusForMediaType_(AVFoundation.AVMediaTypeAudio)
    if status == MIC_NOT_DETERMINED:
        answered = threading.Event()
        device.requestAccessForMediaType_completionHandler_(
            AVFoundation.AVMediaTypeAudio, lambda granted: answered.set())
        answered.wait(timeout=120)
        status = device.authorizationStatusForMediaType_(AVFoundation.AVMediaTypeAudio)
    if status != MIC_AUTHORIZED:
        raise SystemExit(
            "Microphone access is blocked, so the script would only hear silence.\n"
            "Fix: System Settings > Privacy & Security > Microphone > turn on the app you're "
            "running this from (Cursor or Terminal), then quit and reopen that app.\n"
            "If it isn't listed, run:  tccutil reset Microphone com.todesktop.230313mzl4w4u92  "
            "(Cursor), or  tccutil reset Microphone com.apple.Terminal  (Terminal), "
            "reopen the app, and run this again to get the permission prompt."
        )


def current_output_device() -> str:
    result = subprocess.run(
        ["SwitchAudioSource", "-c", "-t", "output"],
        capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


def set_output_device(name: str) -> None:
    subprocess.run(
        ["SwitchAudioSource", "-s", name, "-t", "output"],
        capture_output=True, check=True,
    )


LAST_OUTPUT_FILE = Path(__file__).resolve().parent.parent / ".last_output"


def output_devices() -> list[str]:
    result = subprocess.run(
        ["SwitchAudioSource", "-a", "-t", "output"],
        capture_output=True, text=True, check=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _recover_real_output() -> str:
    """Output was left on BlackHole by a run that never restored it; find the real device."""
    devices = [d for d in output_devices() if d != BLACKHOLE]
    saved = LAST_OUTPUT_FILE.read_text().strip() if LAST_OUTPUT_FILE.exists() else ""
    if saved in devices:
        return saved
    speakers = [d for d in devices if "speakers" in d.lower()]
    if speakers or devices:
        return (speakers or devices)[0]
    raise RuntimeError(f"No output device other than '{BLACKHOLE}' found; pass --output.")


@contextmanager
def system_output_through_blackhole(playback_device: str | None = None):
    """Point macOS output at BlackHole for the duration, then restore it.

    Yields the name of the real device the script should play to.
    """
    original = current_output_device()
    if original == BLACKHOLE:
        original = playback_device or _recover_real_output()
        print(f"System output was left on {BLACKHOLE}; using {original}.")
    LAST_OUTPUT_FILE.write_text(original)
    set_output_device(BLACKHOLE)
    # The macOS volume slider now scales BlackHole, i.e. the audio before the script hears it.
    # At 100% the speakers' own volume (unchanged) sets the loudness, same as without the script.
    subprocess.run(["osascript", "-e", "set volume output volume 100"], capture_output=True)
    try:
        yield playback_device or original
    finally:
        set_output_device(original)
        print(f"Restored system output to {original}.")


def find_device(name: str, kind: str) -> int:
    for index, device in enumerate(sd.query_devices()):
        if device["name"] == name and device[f"max_{kind}_channels"] > 0:
            return index
    raise RuntimeError(f"No {kind} audio device named '{name}'")


class AudioRouter:
    """Copies BlackHole input to the playback device and hands mono blocks to the detector."""

    SAMPLERATE = 48000

    def __init__(self, playback_device: str, samplerate: int = SAMPLERATE, blocksize: int = 512,
                 fade_s: float = 0.05, mute_gain: float = 0.0):
        self.samplerate = samplerate
        self.mute_gain = mute_gain
        self.blocks: queue.Queue[np.ndarray] = queue.Queue(maxsize=200)
        self.dropped_blocks = 0
        self._gain = 1.0
        self._target = 1.0
        self._step = 1.0 / (fade_s * samplerate)
        self._stream = sd.Stream(
            device=(find_device(BLACKHOLE, "input"), find_device(playback_device, "output")),
            samplerate=samplerate,
            blocksize=blocksize,
            channels=2,
            dtype="float32",
            callback=self._callback,
        )

    @property
    def muted(self) -> bool:
        return self._target < 1.0

    def set_muted(self, muted: bool) -> None:
        self._target = self.mute_gain if muted else 1.0

    def __enter__(self):
        self._stream.start()
        return self

    def __exit__(self, *exc):
        self._stream.stop()
        self._stream.close()

    def _callback(self, indata, outdata, frames, time, status):
        gain, target = self._gain, self._target
        if gain == target:
            np.multiply(indata, gain, out=outdata)
        else:
            # Ramp the gain instead of jumping so muting doesn't click.
            direction = 1.0 if target > gain else -1.0
            ramp = gain + direction * self._step * np.arange(1, frames + 1, dtype=np.float32)
            ramp = np.minimum(ramp, target) if direction > 0 else np.maximum(ramp, target)
            np.multiply(indata, ramp[:, None], out=outdata)
            self._gain = float(ramp[-1])

        try:
            self.blocks.put_nowait(indata.mean(axis=1))
        except queue.Full:
            self.dropped_blocks += 1
