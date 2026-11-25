"""Combine the music detector and transcript verdicts into a mute decision."""

from collections import deque

from automute.speech import Verdict


class Arbiter:
    """Treats an ad break as a block: hard to enter, and once in, hard to leave by accident.

    Muting (while live):
    - An unmistakable ad phrase ("MSRP", "visit ...com", "ask your doctor") mutes at once.
    - Music + talking: one AD verdict mutes; two GAME verdicts in a row keep it live (a player
      feature with music under it). With no verdict after `max_wait_s` of music, mute anyway.
    - Talking without music: `speech_mute_count` AD verdicts in a row (a promo with no music).
    - Right after an unmute, two AD verdicts on post-unmute audio re-mute (unmuted too early).
    - Nobody talking: music decides, but a music sting right after game commentary has to
      last `max_wait_s` first.

    Unmuting (while muted): many ads have no music, so music dropping out is not enough.
    - `game_unmute_count` GAME verdicts out of the last `game_unmute_count + 1` on audio from
      after the mute, or two GAME verdicts in a row where one used play-by-play vocabulary, or
    - nobody talking, no music for `quiet_unmute_s`, and no AD verdict in the last 10 s, or
    - transcription has produced nothing for `no_verdict_fallback_s` and the music is gone.
    """

    def __init__(self, hop_s: float, quiet_unmute_s: float = 6.0, max_wait_s: float = 4.0,
                 game_unmute_count: int = 3, speech_mute_count: int = 4,
                 remute_window_s: float = 10.0, verdict_max_age_s: float = 5.0,
                 no_verdict_fallback_s: float = 8.0, music_off: float = 0.15):
        self._recent_music: deque[float] = deque(maxlen=max(1, round(quiet_unmute_s / hop_s)))
        self.max_wait_s = max_wait_s
        self.game_unmute_count = game_unmute_count
        self.speech_mute_count = speech_mute_count
        self.remute_window_s = remute_window_s
        self.verdict_max_age_s = verdict_max_age_s
        self.no_verdict_fallback_s = no_verdict_fallback_s
        self.music_off = music_off
        self._unmuted_at = float("-inf")
        self._muted_at = float("-inf")
        self._last_verdict_at = float("-inf")
        self._last_ad_verdict_at = float("-inf")

    def decide(self, now: float, muted: bool, audio_says_ad: bool, music: float,
               speech_present: bool, verdicts: list[Verdict], music_for_s: float) -> bool:
        self._recent_music.append(music)
        fresh = [v for v in verdicts if now - v.audio_end <= self.verdict_max_age_s]
        if fresh:
            self._last_verdict_at = now
            if fresh[-1].label == "AD":
                self._last_ad_verdict_at = now

        if muted:
            should_mute = self._while_muted(now, speech_present, fresh)
            if not should_mute:
                self._unmuted_at = now
            return should_mute
        should_mute = self._while_live(now, audio_says_ad and music > self.music_off, speech_present,
                                       fresh, music_for_s)
        if should_mute:
            self._muted_at = now
        return should_mute

    def _music_gone(self) -> bool:
        return (len(self._recent_music) == self._recent_music.maxlen
                and max(self._recent_music) < self.music_off)

    def _while_muted(self, now: float, speech_present: bool, fresh: list[Verdict]) -> bool:
        # Transcripts of audio from before the mute say nothing about whether the break is over.
        since_mute = [v for v in fresh if v.audio_start >= self._muted_at - 1.0]
        labels = [v.label for v in since_mute]
        if labels[-self.game_unmute_count:] == ["GAME"] * self.game_unmute_count:
            return False
        # Play-by-play vocabulary ("third and two", "touchdown") backed by one more GAME verdict.
        if labels[-2:] == ["GAME", "GAME"] and any(v.strong for v in since_mute[-2:]):
            return False
        # The LLMs can flip-flop on real commentary; mostly-GAME is enough.
        recent = labels[-(self.game_unmute_count + 1):]
        if len(recent) == self.game_unmute_count + 1 and recent.count("GAME") >= self.game_unmute_count:
            return False
        if not speech_present:
            ad_talk_recently = now - self._last_ad_verdict_at <= 10.0
            return ad_talk_recently or not self._music_gone()
        if now - self._last_verdict_at > self.no_verdict_fallback_s:
            return not self._music_gone()
        return True

    def _while_live(self, now: float, music_now: bool, speech_present: bool,
                    fresh: list[Verdict], music_for_s: float) -> bool:
        labels = [v.label for v in fresh]
        if not speech_present:
            return music_now and ("GAME" not in labels or music_for_s >= self.max_wait_s)
        if music_now:
            if labels and labels[-1] == "AD":
                return True
            if labels[-2:] == ["GAME", "GAME"]:
                return False
            return music_for_s >= self.max_wait_s

        # Only transcripts that are mostly audio from after the last unmute count here;
        # older ones are just the end of the ad that was already muted.
        recent = [v for v in fresh if v.audio_start >= self._unmuted_at - 2.0]
        if recent and recent[-1].label == "AD" and recent[-1].strong:
            return True  # "0% financing", "visit ...com", "ask your doctor": no need to wait
        since_unmute = [v.label for v in recent]
        if since_unmute[-self.speech_mute_count:] == ["AD"] * self.speech_mute_count:
            return True
        recently_unmuted = now - self._unmuted_at <= self.remute_window_s
        return recently_unmuted and since_unmute[-2:] == ["AD", "AD"]
