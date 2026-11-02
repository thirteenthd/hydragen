# SPDX-License-Identifier: Apache-2.0
"""src/hydragen/server/tokenizer.py - Tokenizer, Detokenizer, and Stop String Matcher."""

from __future__ import annotations

import codecs
from typing import Any, List, Optional, Sequence, Tuple


class ByteTokenizer:
    """Tiny stand-in tokenizer for tests without the pack: one id per UTF-8 byte, specials as ids >= 256."""

    SPECIALS = ["<|im_start|>", "<|im_end|>", "<|endoftext|>", "<|vision_start|>", "<|image_pad|>", "<|vision_end|>"]
    ALWAYS: Tuple[str, ...] = ()

    @property
    def max_special_len(self) -> int:
        return max(len(s) for s in self.SPECIALS)

    @property
    def control_tokens(self) -> List[str]:
        return [s for s in self.SPECIALS if s not in self.ALWAYS]

    def encode(self, text: str, parse_special: bool = False, plain: Sequence[Tuple[int, int]] = ()) -> List[int]:
        return self.encode_marked(text, parse_special, plain)[0]

    def encode_marked(
        self,
        text: str,
        parse_special: bool = False,
        plain: Sequence[Tuple[int, int]] = (),
    ) -> Tuple[List[int], List[Tuple[int, int]]]:
        """encode() and its resume points (after each special), as strata_tokenizer's for PromptEncoder."""
        out: List[int] = []
        marks: List[Tuple[int, int]] = []
        i = 0
        while i < len(text):
            for k, s in enumerate(self.SPECIALS):
                if (parse_special or s in self.ALWAYS) and text.startswith(s, i) and not any(
                    a <= i < b for a, b in plain
                ):
                    out.append(256 + k)
                    i += len(s)
                    marks.append((i, len(out)))
                    break
            else:
                out.extend(text[i].encode("utf-8"))
                i += 1
        return out, marks

    def decode(self, ids: Sequence[int], errors: str = "replace") -> str:
        raw = bytearray()
        for t in ids:
            raw += self.SPECIALS[t - 256].encode() if t >= 256 else bytes([t])
        return raw.decode("utf-8", errors=errors)


class Detokenizer:
    """Incremental decode: each token's bytes go through an incremental UTF-8 decoder."""

    def __init__(self, tok: Any):
        self.tok = tok
        self.ids: List[int] = []
        self.sent = 0
        self.inc = codecs.getincrementaldecoder("utf-8")(errors="replace") if hasattr(tok, "token_bytes") else None

    def pending(self) -> bool:
        """A character is split across the tokens so far: its first bytes are held."""
        if self.inc is not None:
            return bool(self.inc.getstate()[0])
        return self.tok.decode(self.ids).endswith("\ufffd")

    def push(self, t: int) -> str:
        if self.inc is not None:
            return self.inc.decode(self.tok.token_bytes(t))
        self.ids.append(t)
        text = self.tok.decode(self.ids)
        if text.endswith("\ufffd"):
            return ""
        delta = text[self.sent:]
        self.sent = len(text)
        return delta

    def flush(self) -> str:
        if self.inc is not None:
            return self.inc.decode(b"", final=True)
        text = self.tok.decode(self.ids)
        delta = text[self.sent:]
        self.sent = len(text)
        return delta


class StopMatcher:
    """Cuts the answer's text at the first stop string."""

    def __init__(self, stops: Sequence[str]):
        self.stops = list(stops)
        self.held = ""
        self.hit: Optional[str] = None

    def push(self, text: str) -> str:
        if self.hit is not None:
            return ""
        buf = self.held + text
        first = None
        for s in self.stops:
            i = buf.find(s)
            if i >= 0 and (first is None or i < first[0]):
                first = (i, s)
        if first is not None:
            self.held, self.hit = "", first[1]
            return buf[:first[0]]
        keep = 0
        for s in self.stops:
            for k in range(min(len(s) - 1, len(buf)), keep, -1):
                if buf.endswith(s[:k]):
                    keep = k
                    break
        self.held = buf[len(buf) - keep:] if keep else ""
        return buf[:len(buf) - keep]

    def flush(self) -> str:
        held, self.held = self.held, ""
        return held


__all__ = ["ByteTokenizer", "Detokenizer", "StopMatcher"]
