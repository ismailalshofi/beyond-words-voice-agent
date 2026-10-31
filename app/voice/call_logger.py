from datetime import datetime
from typing import Literal

ServiceType = Literal["transcription", "llm", "tts", "client", "interruption"]


class CallLogger:
    """Collects timestamped events during a call and prints a per-turn latency breakdown when it ends."""

    def __init__(self):
        self.logs: list[dict] = []

    def add_log(self, service: ServiceType, event: str, content: str = "") -> None:
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        self.logs.append({"service": service, "timestamp": timestamp, "event": event, "content": content})

    def reset(self) -> None:
        self.logs.clear()

    def get_logs(self) -> list[dict]:
        return self.logs.copy()

    def print_summary(self) -> None:
        if not self.logs:
            print("[LOGS] No logs available")
            return

        print("\n" + "=" * 80)
        print("[LOGS_SUMMARY] Call ended")
        print("=" * 80)

        print(f"\n[LOGS] ({len(self.logs)} events)")
        for i, log in enumerate(self.logs, 1):
            service = log["service"].upper()
            print(f"{i:3d}. [{log['timestamp']}] [{service:13s}] [{log['event']:25s}] {log['content']}")

        print("\n" + "=" * 80)
        print("[TIMING_ANALYSIS] Latency breakdown per turn")
        print("=" * 80)
        for idx, turn in enumerate(self._group_turns(), 1):
            metrics = self._extract_metrics(turn)
            if metrics["llm_start"]:
                self._print_turn(idx, metrics)

        print("\nTargets: LLM TTFB < 500ms, TTS first audio < 400ms, end-to-end < 2000ms")
        print("=" * 80 + "\n")

    def _group_turns(self) -> list[list[dict]]:
        """Each LLM call starts a new turn."""
        turns: list[list[dict]] = []
        current: list[dict] = []
        for log in self.logs:
            if log["service"] == "llm" and log["event"] == "call_initiated":
                if current:
                    turns.append(current)
                current = [log]
            elif current or log["service"] == "transcription":
                current.append(log)
        if current:
            turns.append(current)
        return turns

    @staticmethod
    def _extract_metrics(turn: list[dict]) -> dict:
        events = {
            ("transcription", "transcription_complete"): "transcript",
            ("llm", "call_initiated"): "llm_start",
            ("llm", "first_chunk_received"): "llm_first",
            ("llm", "last_chunk_received"): "llm_last",
            ("tts", "first_text_chunk_sent"): "tts_text_first",
            ("tts", "first_audio_received"): "tts_audio_first",
            ("tts", "last_audio_received"): "tts_audio_last",
            ("client", "first_chunk_sent"): "client_first",
            ("client", "last_chunk_sent"): "client_last",
        }
        metrics: dict = {key: None for key in events.values()}
        for log in turn:
            key = events.get((log["service"], log["event"]))
            if key and metrics[key] is None:
                metrics[key] = log["content"] if key == "transcript" else log["timestamp"]
        return metrics

    def _print_turn(self, idx: int, m: dict) -> None:
        print(f"\n{'-' * 80}\nTURN #{idx}")
        if m["transcript"]:
            print(m["transcript"][:80])
        print("-" * 80)
        rows = [
            ("LLM time to first token", m["llm_start"], m["llm_first"], 500, 1000),
            ("LLM total", m["llm_start"], m["llm_last"], None, None),
            ("TTS time to first audio", m["tts_text_first"], m["tts_audio_first"], 400, 800),
            ("TTS total", m["tts_text_first"], m["tts_audio_last"], None, None),
            ("Client streaming", m["client_first"], m["client_last"], None, None),
            ("END-TO-END (response starts)", m["llm_start"], m["client_first"], 2000, 3000),
        ]
        for label, start, end, good, ok in rows:
            duration = self._diff_ms(start, end)
            if duration is None:
                continue
            indicator = ""
            if good and ok:
                indicator = "🔴 " if duration > ok else "⚠️ " if duration > good else "✅ "
            print(f"   {indicator}{label:35s} {duration:8.0f} ms")

    @staticmethod
    def _diff_ms(start: str | None, end: str | None) -> float | None:
        if not start or not end:
            return None
        fmt = "%H:%M:%S.%f"
        return (datetime.strptime(end, fmt) - datetime.strptime(start, fmt)).total_seconds() * 1000
