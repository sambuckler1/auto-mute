"""Auto-mute commercials during sports broadcasts.

Music makes a stretch suspicious; when someone is talking, a Whisper transcript classified
by a local LLM (Ollama) decides whether it's the game or an ad.

Run with:  .venv/bin/python -m automute            (monitor mode: shows scores, you mute by hand)
           .venv/bin/python -m automute --auto     (mutes automatically)
           .venv/bin/python -m automute --auto --gui   (plus a small on-screen monitor)

Hotkeys (work from any app):
  ctrl+option+m   toggle mute by hand (also turns auto mode off)
  ctrl+option+a   toggle auto mode
"""

import argparse
import csv
import logging
import queue
import signal
import threading
import time
import traceback
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
from pynput import keyboard

from automute.audio import AudioRouter, ensure_microphone_access, system_output_through_blackhole
from automute.detector import AdDecider, Windower, YamnetDetector, ensure_model
from automute.decision import Arbiter
from automute.gui import Hud, Telemetry
from automute.speech import SpeechJudge, check_ollama

PROJECT_DIR = Path(__file__).resolve().parent.parent
SILENCE_RMS = 1e-4
HOP_S = 0.5
SPEECH_PRESENT = 0.3        # YAMNet "Speech" score that counts as someone talking
MUSIC_PRESENT = 0.3
LOOKBACK_S = 3.0
VERDICT_MAX_AGE_S = 6.0

# pynput warns "process is not trusted" even when Input Monitoring lets the hotkeys work.
logging.getLogger("pynput").setLevel(logging.ERROR)

LOG_FIELDS = ["time", "rms", "ad", "music", "game", "speech", "smoothed", "auto_says_ad",
              "speech_present", "speech_verdict", "final_ad", "muted", "mode", "top", "transcript"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mute commercials by detecting their music.")
    parser.add_argument("--auto", action="store_true", help="start with automatic muting on")
    parser.add_argument("--gui", action="store_true", help="show a small always-on-top monitor window")
    parser.add_argument("--output", help="device to play through (default: current system output)")
    parser.add_argument("--ad-above", type=float, default=0.35, help="smoothed ad score above this means ad")
    parser.add_argument("--game-below", type=float, default=0.15, help="smoothed ad score below this means game")
    parser.add_argument("--ad-after", type=float, default=1.5, help="seconds above --ad-above before muting")
    parser.add_argument("--game-after", type=float, default=1.5,
                        help="seconds below --game-below before the music detector calls it game")
    parser.add_argument("--unmute-after", type=float, default=6.0,
                        help="seconds with no talking and no music before unmuting")
    parser.add_argument("--game-unmute-count", type=int, default=3,
                        help="GAME transcripts in a row needed to unmute")
    parser.add_argument("--smooth", type=float, default=3.0, help="seconds of scores to average")
    parser.add_argument("--duck", type=float, default=0.0, help="volume during ads, 0 = silent, 0.2 = quiet")
    parser.add_argument("--no-log", action="store_true", help="don't write a CSV log to logs/")
    parser.add_argument("--no-speech", action="store_true", help="decide from music alone, no transcription")
    parser.add_argument("--whisper-model", default="mlx-community/whisper-small.en-mlx")
    parser.add_argument("--llm-models", default="qwen2.5:7b,llama3.2",
                        help="comma-separated Ollama models; they must agree to change state")
    return parser.parse_args()


class Controller:
    """Holds the mute/auto state shared between the audio loop and the hotkey thread."""

    def __init__(self, router: AudioRouter, auto: bool):
        self.router = router
        self.auto = auto
        self._lock = threading.Lock()

    @property
    def mode(self) -> str:
        return "auto" if self.auto else "manual"

    def toggle_mute(self) -> None:
        with self._lock:
            self.auto = False
            self.router.set_muted(not self.router.muted)
            announce(f"{'muted' if self.router.muted else 'unmuted'} by hand (auto mode off)")

    def toggle_auto(self) -> None:
        with self._lock:
            self.auto = not self.auto
            announce(f"auto mode {'on' if self.auto else 'off'}")

    def apply_detection(self, is_ad: bool) -> None:
        with self._lock:
            if self.auto and is_ad != self.router.muted:
                self.router.set_muted(is_ad)
                announce("ad detected, muted" if is_ad else "game is back, unmuted")


def announce(message: str) -> None:
    print(f"\r\033[K[{datetime.now():%H:%M:%S}] {message}", flush=True)


def bar(value: float, width: int = 10) -> str:
    filled = int(round(min(max(value, 0.0), 1.0) * width))
    return "█" * filled + "·" * (width - filled)


def open_log(disabled: bool, class_labels: list[str]):
    """One row per window; the per-class YAMNet scores are the training data for a classifier."""
    if disabled:
        return None, None
    log_dir = PROJECT_DIR / "logs"
    log_dir.mkdir(exist_ok=True)
    path = log_dir / f"session-{datetime.now():%Y%m%d-%H%M%S}.csv"
    f = open(path, "w", newline="")
    writer = csv.DictWriter(f, fieldnames=LOG_FIELDS + class_labels)
    writer.writeheader()
    print(f"Logging to {path.relative_to(PROJECT_DIR)}")
    return f, writer


def run(args: argparse.Namespace, stop: threading.Event) -> None:
    ensure_microphone_access()
    model_path, class_map_path = ensure_model(PROJECT_DIR / "models")
    detector = YamnetDetector(model_path, class_map_path)
    decider = AdDecider(args.ad_above, args.game_below, args.ad_after, args.game_after, args.smooth, HOP_S)
    arbiter = Arbiter(HOP_S, quiet_unmute_s=args.unmute_after, game_unmute_count=args.game_unmute_count)
    speech = None
    if not args.no_speech:
        print("Loading speech models (Whisper + Ollama)...")
        llm_models = [m.strip() for m in args.llm_models.split(",") if m.strip()]
        check_ollama(llm_models)
        speech = SpeechJudge(AudioRouter.SAMPLERATE, args.whisper_model, llm_models)
        speech.start()
        speech.ready.wait()
        if speech.load_error:
            raise SystemExit(f"Couldn't load Whisper ({speech.load_error!r}). Run with --no-speech to skip it.")
    log_file, log = open_log(args.no_log, detector.labels)

    with system_output_through_blackhole(args.output) as playback_device:
        router = AudioRouter(playback_device, mute_gain=args.duck)
        controller = Controller(router, auto=args.auto)
        windower = Windower(router.samplerate, HOP_S)
        telemetry = Telemetry(mode=controller.mode)

        def process_audio() -> None:
            recent_speech: deque[float] = deque(maxlen=round(LOOKBACK_S / HOP_S))
            recent_music: deque[float] = deque(maxlen=round(LOOKBACK_S / HOP_S))
            music_since = None
            last_verdict = None

            while not stop.is_set():
                try:
                    block = router.blocks.get(timeout=0.2)
                except queue.Empty:
                    continue
                if speech:
                    speech.push(block)
                window = windower.push(block)
                if window is None:
                    continue

                now = time.monotonic()
                rms = float(np.sqrt(np.mean(window ** 2)))
                row = {"time": datetime.now().isoformat(timespec="milliseconds"), "rms": f"{rms:.5f}"}

                verdicts = speech.recent_verdicts(VERDICT_MAX_AGE_S) if speech else []
                labels = [v.label for v in verdicts]
                if speech and speech.latest is not last_verdict:
                    if speech.latest and (last_verdict is None or speech.latest.label != last_verdict.label):
                        announce(f'speech sounds like {speech.latest.label}: "{speech.latest.text[:90]}"')
                    last_verdict = speech.latest
                    row["transcript"] = last_verdict.text if last_verdict else ""
                    if last_verdict:
                        telemetry.transcript, telemetry.transcript_label = last_verdict.text, last_verdict.label

                if rms < SILENCE_RMS:
                    decider.hold()
                    controller.apply_detection(arbiter.decide(
                        now, router.muted, decider.is_ad, 0.0, False, verdicts, 0.0))
                    status = "silent (nothing playing, or Microphone permission missing)"
                    telemetry.silent = True
                    telemetry.ad = telemetry.music = telemetry.crowd = telemetry.speech = 0.0
                else:
                    result = detector.analyze(window)
                    audio_says_ad = decider.update(result["ad"], now)
                    if not audio_says_ad:
                        music_since = None
                    elif music_since is None:
                        music_since = now
                    recent_speech.append(result["speech"])
                    recent_music.append(result["music"])
                    speech_present = speech is not None and max(recent_speech) > SPEECH_PRESENT
                    if speech:
                        speech.music_playing = max(recent_music) > MUSIC_PRESENT
                        speech.in_break = router.muted
                    final_ad = arbiter.decide(now, router.muted, audio_says_ad, result["music"],
                                              speech_present, verdicts,
                                              now - music_since if music_since else 0.0)
                    controller.apply_detection(final_ad)

                    talk = labels[-1] if labels else ("..." if speech_present else "-")
                    status = (f"ad {bar(result['ad'])} {result['ad']:.2f}  "
                              f"avg {decider.smoothed:.2f}  music {result['music']:.2f}  "
                              f"crowd {result['game']:.2f}  talk {talk:4}  {result['top'][:20]}")
                    row.update({k: f"{result[k]:.3f}" for k in ("ad", "music", "game", "speech")},
                               top=result["top"], smoothed=f"{decider.smoothed:.3f}",
                               speech_present=int(speech_present), final_ad=int(final_ad),
                               speech_verdict=labels[-1] if labels else "")
                    row.update(zip(detector.labels, (f"{s:.3f}" for s in result["scores"])))
                    telemetry.silent = False
                    telemetry.ad, telemetry.music = result["ad"], result["music"]
                    telemetry.crowd, telemetry.speech = result["game"], result["speech"]
                    telemetry.top, telemetry.talk = result["top"], talk

                telemetry.smoothed = decider.smoothed
                telemetry.muted, telemetry.mode = router.muted, controller.mode
                telemetry.history.append((telemetry.ad, router.muted))

                state = "MUTED" if router.muted else "live "
                print(f"\r\033[K{controller.mode:6} {state} | {status}", end="", flush=True)

                if log:
                    row.update(auto_says_ad=int(decider.is_ad), muted=int(router.muted),
                               mode=controller.mode)
                    log.writerow(row)
                    log_file.flush()

        def process_audio_until_error() -> None:
            try:
                process_audio()
            except Exception:
                traceback.print_exc()
                stop.set()

        hotkeys = keyboard.GlobalHotKeys({
            "<ctrl>+<alt>+m": controller.toggle_mute,
            "<ctrl>+<alt>+a": controller.toggle_auto,
        })
        hotkeys.start()

        print(f"Playing through: {playback_device}  (system output is now BlackHole)")
        print(f"Mode: {controller.mode}.  ctrl+option+m = mute,  ctrl+option+a = auto,  ctrl+c = quit\n")

        with router:
            try:
                if args.gui:
                    # macOS only lets the main thread run the GUI, so audio moves to a worker.
                    worker = threading.Thread(target=process_audio_until_error, daemon=True)
                    worker.start()
                    Hud(telemetry, args.ad_above, args.game_below,
                        controller.toggle_mute, controller.toggle_auto, stop).run()
                    stop.set()
                    worker.join(timeout=5)
                else:
                    process_audio()
            finally:
                hotkeys.stop()
                if speech:
                    speech.stop()
                if log_file:
                    log_file.close()
                print()


def main() -> None:
    args = parse_args()
    # Raising from a signal handler can abort cleanup midway and leave the system
    # output on BlackHole, so signals only ask the main loop to stop.
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, lambda *_: stop.set())
    run(args, stop)


if __name__ == "__main__":
    main()
