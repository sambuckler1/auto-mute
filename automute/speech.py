"""Transcribe recent speech with Whisper and ask a local LLM whether it's the game or an ad."""

import json
import re
import threading
import time
import urllib.request
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
from scipy.signal import resample_poly

WHISPER_RATE = 16000
OLLAMA_URL = "http://localhost:11434/api/generate"

PROMPT = """You are listening to a live football broadcast on TV. Below are a few seconds of
transcribed speech (it may start or end mid-sentence). Decide where it came from.

GAME = the live broadcast of the game being watched. Unscripted, conversational commentators
reacting to what is happening right now: downs and distance, plays, penalties, replays,
challenges, injuries, the score, the clock, what a player just did, analysts talking to each other.

AD = anything scripted that interrupts the game: commercials, and promos for other games,
shows, or networks. Signs: polished narration or slogans, talking to the viewer ("you", "your"),
brands, apps, sportsbooks, products, offers, websites, dates and times ("Saturday at noon",
"tonight at 8"), network names ("on Fox", "on CBS", "only on Peacock"), legal fine print.
Football-themed ads and promos for OTHER games are very common - they are still AD.

Examples:
"second and six, he fakes the handoff, rolls right, and throws it away" -> GAME
"that's a hold on the left tackle, you can see him grab the jersey right there" -> GAME
"he's been their best receiver all year, Tony, just a matchup nightmare" -> GAME
"Michigan travels to Penn State with the Big Ten title on the line, Saturday at 3:30 on CBS" -> AD
"no matter where game day takes you, we'll be there" -> AD
"Bet $5 and get $250 in bonus bets if your team wins. Must be 21 plus" -> AD
"the ultimate tailgate starts with an ice cold" -> AD
{music_note}
Answer with exactly one word: GAME or AD.

Transcript: "{text}"
"""

MUSIC_NOTE = "There is music playing under this speech, which is more common in ads and promos.\n"
# The prompt deliberately says nothing about whether we're currently muted: telling the model
# "we're in a break" made it echo the current state (25% of transcripts flipped on that alone).


# Phrases that settle the question without asking an LLM. Commentators essentially never say
# the ad ones, and ad copy rarely uses play-by-play vocabulary (promos for other games mention
# kickoffs, but also say "on Fox"/"at noon", so mixed transcripts go to the LLMs instead).
# Sponsor phrases ("presented by", "brought to you by") are left to the LLMs: commentators read
# them during the game ("first half stats are brought to you by Little Caesars").
AD_CUES = re.compile(
    r"\b(msrp|apr|financing|lease (a|the|for)|per month|a month for|dot com|\w\.com|"
    r"visit (?!(with|and)\b)\w+|"
    r"download (the|our)|app store|promo code|terms (and conditions )?apply|restrictions apply|"
    r"limited time|offer ends|while supplies last|ask your doctor|side effects|call (now|today)|"
    r"order (now|today)|sportsbook|bonus bets|must be 21|participating|available (now|at)|"
    r"only on (fox|cbs|nbc|espn|abc|peacock|prime|netflix|paramount)|streaming (now|on)|"
    r"(on|at) (fox|cbs|nbc|espn|abc) (this|next|sunday|saturday|monday|tonight)|"
    r"(saturday|sunday|monday|thursday|tonight|tomorrow) on (fox|cbs|nbc|espn|abc|peacock|prime)|"
    r"\d+ ?(%|percent) off|save up to|starting at \$?\d)\b",
    re.IGNORECASE)
GAME_CUES = re.compile(
    r"\b((first|second|third|fourth) (and|down)\b|yard line|\d+[- ]yard|touchdown|field goal|"
    r"flag (is )?down|penalty|incomplete|sacked|punt|intercept|fumble|"
    r"(first|second|third|fourth) quarter|two[- ]minute warning|shotgun|red zone|end zone|"
    r"line of scrimmage|return from|blitz|extra point|holding|offsides|false start|"
    r"replay shows|the call (on the field )?stands)",
    re.IGNORECASE)


def transcript_cue(text: str) -> str | None:
    ad, game = AD_CUES.search(text), GAME_CUES.search(text)
    if ad and not game:
        return "AD"
    if game and not ad:
        return "GAME"
    return None


@dataclass
class Verdict:
    label: str            # "GAME" or "AD"
    text: str
    audio_start: float    # time.monotonic() span of the audio that was transcribed
    audio_end: float
    strong: bool = False  # decided by an unmistakable phrase rather than the LLMs


def classify_transcript(text: str, model: str, music: bool = False, timeout_s: float = 10) -> str | None:
    body = json.dumps({
        "model": model,
        "prompt": PROMPT.format(text=text, music_note=MUSIC_NOTE if music else ""),
        "stream": False,
        "keep_alive": "30m",
        "options": {"temperature": 0, "num_predict": 3},
    }).encode()
    request = urllib.request.Request(OLLAMA_URL, body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        answer = json.load(response)["response"].strip().upper()
    if answer.startswith("AD"):
        return "AD"
    if answer.startswith("GAME"):
        return "GAME"
    return None


def classify_with_panel(text: str, models: list[str], music: bool) -> tuple[str | None, str]:
    """Ask several models at once and return (label, reason); label is None unless they all agree.

    A split panel means the transcript is ambiguous, so it produces no verdict rather than a vote
    for either side (previously a split counted as whichever label kept the current state).
    """
    with ThreadPoolExecutor(max_workers=len(models)) as pool:
        answers = list(pool.map(lambda m: classify_transcript(text, m, music), models))
    if None in answers:
        return None, "llm_unparseable"
    if len(set(answers)) > 1:
        return None, "panel_split"
    return answers[0], ""


def check_ollama(models: list[str]) -> None:
    for model in models:
        try:
            # The first request loads the model into memory, which can take a while.
            classify_transcript("Second and seven from the forty.", model, timeout_s=120)
        except Exception as error:
            raise RuntimeError(
                f"Couldn't reach Ollama model '{model}' ({error}). Open the Ollama app or run "
                f"`ollama serve`, and `ollama pull {model}` if needed. Or run with --no-speech."
            ) from error


class SpeechJudge:
    """Background thread: every `interval_s`, transcribe the last `window_s` of audio and classify it.

    A short stretch of near-silence (the usual gap between an ad and the game, or between ads)
    starts a new segment: older verdicts are dropped and transcripts only cover audio after it,
    so the first verdict after a break isn't polluted by the tail of the previous one.
    """

    # Filters for Whisper "hallucinations": over music or noise it tends to invent short phrases
    # or loop one token ("a a a a ..."), which shows up as highly compressible, low-confidence text.
    MIN_WORDS = 5
    NO_SPEECH_PROB = 0.6
    MAX_COMPRESSION_RATIO = 2.4
    MIN_AVG_LOGPROB = -1.0
    MIN_UNIQUE_WORD_RATIO = 0.4

    GAP_RMS = 0.002          # about -54 dBFS
    GAP_S = 0.3
    MIN_SEGMENT_S = 1.5      # don't transcribe less audio than this after a gap

    def __init__(self, input_rate: int, whisper_model: str, llm_models: list[str],
                 window_s: float = 4.0, interval_s: float = 0.75):
        import mlx_whisper  # slow import; keep it out of --no-speech runs

        self._transcribe = mlx_whisper.transcribe
        self._whisper_model = whisper_model
        self._llm_models = llm_models
        self._input_rate = input_rate
        self._window_len = int(window_s * input_rate)
        self._interval_s = interval_s
        self._ring = np.zeros(self._window_len, dtype=np.float32)
        self._write_pos = 0
        self._buffer_end = 0.0
        self._silent_run = 0
        self._since_gap = self._window_len
        self._segment = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.ready = threading.Event()
        self.load_error: Exception | None = None
        self.latest: Verdict | None = None
        self._verdicts: deque[Verdict] = deque(maxlen=8)
        self.music_playing = False      # set by the caller; passed to the LLM as a hint
        self.errors = 0
        self._drops: list[str] = []     # why recent attempts produced no verdict; see take_drops()
        self._drops_lock = threading.Lock()  # separate from _lock, which push() holds when it drops

    def push(self, block: np.ndarray) -> None:
        with self._lock:
            end = self._write_pos + len(block)
            if end <= self._window_len:
                self._ring[self._write_pos:end] = block
            else:
                split = self._window_len - self._write_pos
                self._ring[self._write_pos:] = block[:split]
                self._ring[:len(block) - split] = block[split:]
            self._write_pos = end % self._window_len
            self._buffer_end = time.monotonic()

            if np.sqrt(np.mean(block ** 2)) < self.GAP_RMS:
                self._silent_run += len(block)
            else:
                if self._silent_run >= self.GAP_S * self._input_rate:
                    self._since_gap = 0
                    self._segment += 1
                    if self._verdicts:
                        self._drop("gap_cleared_verdicts")
                    self._verdicts.clear()
                self._silent_run = 0
            self._since_gap = min(self._since_gap + len(block), self._window_len)

    def _snapshot(self) -> tuple[np.ndarray, float, int]:
        with self._lock:
            audio = np.concatenate((self._ring[self._write_pos:], self._ring[:self._write_pos]))
            return audio[-self._since_gap:], self._buffer_end, self._segment

    def _drop(self, reason: str) -> None:
        """Record why an attempt produced no verdict."""
        with self._drops_lock:
            self._drops.append(reason)

    def take_drops(self) -> list[str]:
        """Drop reasons since the last call, for the CSV log."""
        with self._drops_lock:
            drops, self._drops = self._drops, []
        return drops

    def recent_verdicts(self, max_age_s: float) -> list[Verdict]:
        """Verdicts on audio from the last `max_age_s` seconds, oldest first."""
        cutoff = time.monotonic() - max_age_s
        return [v for v in list(self._verdicts) if v.audio_end >= cutoff]

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    def _run(self) -> None:
        # MLX state is per-thread, so Whisper has to be loaded on the thread that uses it.
        try:
            self._transcribe(np.zeros(WHISPER_RATE, dtype=np.float32), path_or_hf_repo=self._whisper_model)
        except Exception as error:
            self.load_error = error
            return
        finally:
            self.ready.set()
        while not self._stop.is_set():
            started = time.monotonic()
            try:
                self._judge_once()
            except Exception as error:
                if self.errors == 0:
                    print(f"\nSpeech check failed ({error!r}); falling back to music only until it recovers."
                          " Later failures are only written to the log (speech_dropped column).")
                self.errors += 1
                self._drop(f"error:{type(error).__name__}")
            self._stop.wait(max(0.0, self._interval_s - (time.monotonic() - started)))

    def _judge_once(self) -> None:
        audio, audio_end, segment = self._snapshot()
        if np.sqrt(np.mean(audio ** 2)) < 1e-4:
            return  # silence; not worth logging
        if len(audio) < self.MIN_SEGMENT_S * self._input_rate:
            self._drop("too_soon_after_gap")
            return
        audio16k = resample_poly(audio, WHISPER_RATE, self._input_rate).astype(np.float32)
        result = self._transcribe(audio16k, path_or_hf_repo=self._whisper_model, language="en",
                                  condition_on_previous_text=False)
        raw = " ".join(segment["text"].strip() for segment in result["segments"]).strip()
        text = " ".join(segment["text"].strip() for segment in result["segments"]
                        if segment["no_speech_prob"] < self.NO_SPEECH_PROB
                        and segment["compression_ratio"] <= self.MAX_COMPRESSION_RATIO
                        and segment["avg_logprob"] >= self.MIN_AVG_LOGPROB).strip()
        words = text.lower().split()
        if not words:
            self._drop(f"whisper_filtered:{raw[:60]}" if raw else "whisper_empty")
            return
        if len(words) < self.MIN_WORDS:
            self._drop(f"too_few_words:{text[:60]}")
            return
        if len(set(words)) / len(words) < self.MIN_UNIQUE_WORD_RATIO:
            self._drop(f"repetitive:{text[:60]}")
            return
        cue = transcript_cue(text)
        if cue:
            label, reason = cue, ""
        else:
            label, reason = classify_with_panel(text, self._llm_models, self.music_playing)
        if not label:
            self._drop(f"{reason}:{text[:60]}")
            return
        verdict = Verdict(label, text, audio_end - len(audio) / self._input_rate, audio_end,
                          strong=cue is not None)
        with self._lock:
            if segment != self._segment:
                self._drop("stale_segment")  # a gap started a new segment while this was being transcribed
                return
            self.latest = verdict
            self._verdicts.append(verdict)
