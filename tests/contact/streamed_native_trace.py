"""Bounded-memory reading of native trace JSON, including gzip integrity."""
import gzip
import json
from pathlib import Path


class _Reader:
    def __init__(self, stream, chunk_size=65536, max_token=16 * 1024 * 1024):
        self.stream, self.chunk_size, self.max_token = stream, chunk_size, max_token
        self.buffer, self.pos, self.eof = "", 0, False
        self.decoder = json.JSONDecoder()

    def _fill(self):
        self.buffer = self.buffer[self.pos:]
        self.pos = 0
        piece = self.stream.read(self.chunk_size)
        if piece:
            self.buffer += piece
        else:
            self.eof = True
        if len(self.buffer) > self.max_token:
            raise ValueError("native trace JSON token exceeds bounded reader limit")

    def _space(self):
        while True:
            while self.pos < len(self.buffer) and self.buffer[self.pos] in " \t\r\n":
                self.pos += 1
            if self.pos < len(self.buffer) or self.eof:
                return
            self._fill()

    def peek(self):
        self._space()
        return self.buffer[self.pos] if self.pos < len(self.buffer) else ""

    def expect(self, char):
        if self.peek() != char:
            raise ValueError("malformed native trace JSON: expected " + repr(char))
        self.pos += 1

    def value(self, delimiters):
        self._space()
        while True:
            try:
                value, end = self.decoder.raw_decode(self.buffer, self.pos)
            except json.JSONDecodeError as exc:
                if self.eof:
                    raise ValueError("malformed or truncated native trace JSON") from exc
                self._fill()
                continue
            # A number at the current buffer boundary may have more digits.
            if end == len(self.buffer) and not self.eof:
                self._fill()
                continue
            cursor = end
            while cursor < len(self.buffer) and self.buffer[cursor] in " \t\r\n":
                cursor += 1
            if cursor == len(self.buffer) and not self.eof:
                self._fill()
                continue
            if cursor == len(self.buffer) or self.buffer[cursor] not in delimiters:
                if not self.eof:
                    self._fill()
                    continue
                raise ValueError("invalid delimiter in native trace JSON")
            self.pos = end
            return value


def read_native(path, on_trace=None, trace_key="traces"):
    """Read all root fields; visit each trace once without retaining the array.

    `on_trace(metadata_so_far, row, index)` is called in file order. The file
    must be complete, have one trace array, and end after the root object.
    """
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        r = _Reader(stream)
        data, count, saw_trace = {}, 0, False
        r.expect("{")
        while r.peek() != "}":
            key = r.value(":")
            if not isinstance(key, str) or key in data or (key == trace_key and saw_trace):
                raise ValueError("invalid or duplicate native trace root key")
            r.expect(":")
            if key == trace_key:
                saw_trace = True
                r.expect("[")
                while r.peek() != "]":
                    row = r.value(",]")
                    if on_trace is not None:
                        on_trace(data, row, count)
                    count += 1
                    if r.peek() == ",":
                        r.expect(",")
                        if r.peek() == "]": raise ValueError("trailing trace comma")
                    elif r.peek() != "]":
                        raise ValueError("invalid trace array separator")
                r.expect("]")
            else:
                data[key] = r.value(",}")
            if r.peek() == ",":
                r.expect(",")
                if r.peek() == "}": raise ValueError("trailing root comma")
            elif r.peek() != "}":
                raise ValueError("invalid root separator")
        r.expect("}")
        if r.peek(): raise ValueError("trailing data after native trace")
        if not saw_trace: raise ValueError("native trace array is missing")
    return data, count
