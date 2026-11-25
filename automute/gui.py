"""Tiny always-on-top HUD showing what the detector is hearing.

Drag it anywhere. Click the LIVE/MUTED readout to toggle mute, the AUTO/MANUAL tag to
toggle auto mode, and x to quit.
"""

import threading
import time
import tkinter as tk
from collections import deque
from dataclasses import dataclass, field

WIDTH, HEIGHT = 300, 214
FONT = "Menlo"

BG = "#04070b"
PANEL = "#081018"
GRID = "#0d2229"
DIM = "#3b6670"
TEXT = "#9fe8f0"
CYAN = "#00e5ff"
GREEN = "#39ff88"
PINK = "#ff2bd6"
AMBER = "#ffb000"
RED_BG = "#2a0a24"

GAME_COLOR = GREEN
AD_COLOR = PINK


@dataclass
class Telemetry:
    """Written by the audio loop, read by the HUD."""
    muted: bool = False
    mode: str = "auto"
    silent: bool = True
    ad: float = 0.0
    smoothed: float = 0.0
    music: float = 0.0
    crowd: float = 0.0
    speech: float = 0.0
    top: str = ""
    talk: str = "-"
    transcript: str = ""
    transcript_label: str = ""
    history: deque = field(default_factory=lambda: deque(maxlen=120))  # (ad score, muted), 60 s


class Hud:
    def __init__(self, telemetry: Telemetry, ad_above: float, game_below: float,
                 on_toggle_mute, on_toggle_auto, stop: threading.Event):
        self.t = telemetry
        self.ad_above, self.game_below = ad_above, game_below
        self.on_toggle_mute, self.on_toggle_auto = on_toggle_mute, on_toggle_auto
        self.stop = stop
        self.started = time.monotonic()

        self.root = tk.Tk()
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        self.root.attributes("-alpha", 0.93)
        x = self.root.winfo_screenwidth() - WIDTH - 14
        self.root.geometry(f"{WIDTH}x{HEIGHT}+{x}+38")

        self.canvas = tk.Canvas(self.root, width=WIDTH, height=HEIGHT, bg=BG,
                                highlightthickness=1, highlightbackground=CYAN)
        self.canvas.pack()
        self._draw_static()

        self._press = None
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)

    def run(self) -> None:
        self._tick()
        self.root.mainloop()

    # ---- drawing ---------------------------------------------------------------------------

    def _draw_static(self) -> None:
        c = self.canvas
        for y in range(0, HEIGHT, 3):
            c.create_line(0, y, WIDTH, y, fill="#060b10")
        c.create_text(10, 11, text="◢ AUTO-MUTE", anchor="w", fill=CYAN, font=(FONT, 10, "bold"))
        c.create_text(98, 11, text="// SIGNAL MONITOR", anchor="w", fill=DIM, font=(FONT, 8))
        c.create_text(WIDTH - 10, 11, text="×", anchor="e", fill=DIM, font=(FONT, 11), tags="close")
        c.create_line(8, 22, WIDTH - 8, 22, fill=GRID)
        for i, name in enumerate(("MUSIC", "CROWD", "VOICE", "AD·AVG")):
            c.create_text(10, 70 + i * 13, text=name, anchor="w", fill=DIM, font=(FONT, 8))
        c.create_rectangle(8, 122, WIDTH - 8, 162, outline=GRID, fill=PANEL)
        c.create_text(12, 127, text="AD SCORE · 60s", anchor="w", fill=DIM, font=(FONT, 7))
        c.create_line(8, 170, WIDTH - 8, 170, fill=GRID)

    def _tick(self) -> None:
        if self.stop.is_set():
            self.root.destroy()
            return
        self.canvas.delete("dyn")
        self._draw_state()
        self._draw_meters()
        self._draw_graph()
        self._draw_transcript()
        self.root.after(100, self._tick)

    def _draw_state(self) -> None:
        c, t = self.canvas, self.t
        blink = int(time.monotonic() * 2) % 2 == 0
        if t.muted:
            color, label, sub = AD_COLOR, "■ MUTED", "AD BREAK DETECTED"
        elif t.silent:
            color, label, sub = DIM, "◌ IDLE", "NO SIGNAL"
        else:
            color, label, sub = GAME_COLOR, "● LIVE", "GAME FEED"
        c.create_text(10, 40, text=label, anchor="w", fill=color if blink or not t.muted else "#a01c88",
                      font=(FONT, 18, "bold"), tags=("dyn", "state"))
        c.create_text(12, 57, text=sub, anchor="w", fill=color, font=(FONT, 7), tags="dyn")

        mode = t.mode.upper()
        mode_color = CYAN if t.mode == "auto" else AMBER
        c.create_rectangle(WIDTH - 82, 30, WIDTH - 10, 46, outline=mode_color, tags=("dyn", "mode"))
        c.create_text(WIDTH - 46, 38, text=mode, fill=mode_color, font=(FONT, 8, "bold"), tags=("dyn", "mode"))
        uptime = int(time.monotonic() - self.started)
        c.create_text(WIDTH - 10, 56, text=f"T+{uptime // 3600:02d}:{uptime // 60 % 60:02d}:{uptime % 60:02d}",
                      anchor="e", fill=DIM, font=(FONT, 7), tags="dyn")

    def _draw_meters(self) -> None:
        values = (
            (self.t.music, AD_COLOR),
            (self.t.crowd, GAME_COLOR),
            (self.t.speech, CYAN),
            (self.t.smoothed, AMBER),
        )
        segments, x0, seg_w = 22, 64, 8
        for i, (value, color) in enumerate(values):
            y = 70 + i * 13
            lit = round(min(max(value, 0.0), 1.0) * segments)
            for s in range(segments):
                x = x0 + s * seg_w
                self.canvas.create_rectangle(x, y - 4, x + seg_w - 2, y + 4, width=0,
                                             fill=color if s < lit else GRID, tags="dyn")
            self.canvas.create_text(WIDTH - 10, y, text=f"{value:.2f}", anchor="e", fill=TEXT,
                                    font=(FONT, 8), tags="dyn")

    def _draw_graph(self) -> None:
        c = self.canvas
        left, right, top, bottom = 10, WIDTH - 10, 132, 160
        history = list(self.t.history)
        step = (right - left) / (self.t.history.maxlen - 1)
        offset = self.t.history.maxlen - len(history)

        for i, (_, muted) in enumerate(history):
            if muted:
                x = left + (offset + i) * step
                c.create_rectangle(x, top, x + step + 1, bottom, width=0, fill=RED_BG, tags="dyn")
        for threshold, color in ((self.ad_above, AD_COLOR), (self.game_below, GAME_COLOR)):
            y = bottom - threshold * (bottom - top)
            c.create_line(left, y, right, y, fill=color, dash=(2, 4), tags="dyn")
        if len(history) >= 2:
            points = []
            for i, (score, _) in enumerate(history):
                points += [left + (offset + i) * step, bottom - min(score, 1.0) * (bottom - top)]
            c.create_line(*points, fill=CYAN, width=1.5, tags="dyn")

    def _draw_transcript(self) -> None:
        c, t = self.canvas, self.t
        label = t.transcript_label or "--"
        color = AD_COLOR if label == "AD" else GAME_COLOR if label == "GAME" else DIM
        c.create_text(10, 181, text="VOX", anchor="w", fill=DIM, font=(FONT, 7), tags="dyn")
        c.create_text(32, 181, text=f"▸ {label}", anchor="w", fill=color, font=(FONT, 8, "bold"), tags="dyn")
        c.create_text(WIDTH - 10, 181, text=(t.top or "")[:22].upper(), anchor="e", fill=DIM,
                      font=(FONT, 7), tags="dyn")

        text = f'"{t.transcript}"' if t.transcript else "awaiting speech..."
        width_chars = 44
        if len(text) > width_chars:
            loop = text + "   ///   "
            start = int(time.monotonic() * 6) % len(loop)
            text = (loop + loop)[start:start + width_chars]
        c.create_text(10, 199, text=text, anchor="w", fill=TEXT, font=(FONT, 8), tags="dyn")

    # ---- interaction -----------------------------------------------------------------------

    def _on_press(self, event) -> None:
        self._press = (event.x_root, event.y_root, self.root.winfo_x(), self.root.winfo_y(), event.x, event.y)

    def _on_drag(self, event) -> None:
        if self._press:
            x_root, y_root, win_x, win_y, *_ = self._press
            self.root.geometry(f"+{win_x + event.x_root - x_root}+{win_y + event.y_root - y_root}")

    def _on_release(self, event) -> None:
        if not self._press:
            return
        x_root, y_root, _, _, x, y = self._press
        self._press = None
        if abs(event.x_root - x_root) + abs(event.y_root - y_root) > 3:
            return  # it was a drag
        tags = set()
        for item in self.canvas.find_overlapping(x - 2, y - 2, x + 2, y + 2):
            tags.update(self.canvas.gettags(item))
        if "close" in tags:
            self.stop.set()
        elif "mode" in tags:
            self.on_toggle_auto()
        elif "state" in tags:
            self.on_toggle_mute()
