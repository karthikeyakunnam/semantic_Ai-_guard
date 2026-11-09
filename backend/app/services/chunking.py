import re
from dataclasses import asdict, dataclass

@dataclass(frozen=True)
class Passage:
    text: str
    start_char: int
    end_char: int
    excluded: bool = False
    exclusion_reason: str | None = None

    def as_dict(self):
        return asdict(self)

def chunk_passages(text: str, max_chars: int = 900) -> list[Passage]:
    """Chunk extracted text while retaining [start,end) offsets into the exact input."""
    result = []
    boundary = re.compile(r"\n+|(?<=[.!?])\s+(?=[A-Z0-9])")
    for match in boundary.finditer(text):
        pass
    starts = [0] + [m.end() for m in boundary.finditer(text)]
    ends = [m.start() for m in boundary.finditer(text)] + [len(text)]
    spans = []
    for start, end in zip(starts, ends):
        while start < end and text[start].isspace(): start += 1
        while end > start and text[end - 1].isspace(): end -= 1
        if end - start > 0:
            spans.append((start, end))
    for start, end in spans:
        if end - start > max_chars:
            pos = start
            while pos < end:
                stop = min(pos + max_chars, end)
                if stop < end:
                    split = text.rfind(" ", pos, stop)
                    if split > pos + 1: stop = split
                left, right = pos, stop
                while left < right and text[left].isspace(): left += 1
                while right > left and text[right - 1].isspace(): right -= 1
                if right - left > 0:
                    result.append(Passage(re.sub(r"\s+", " ", text[left:right]).strip(), left, right))
                pos = max(stop, pos + 1)
        else:
            result.append(Passage(re.sub(r"\s+", " ", text[start:end]).strip(), start, end))
    return result

def split_passages(text: str, max_chars: int = 900) -> list[str]:
    return [passage.text for passage in chunk_passages(text, max_chars)]
