import json, re, xml.etree.ElementTree as ET
from typing import Optional, List, Dict

def strip_fences_and_maybe_think(text: str, allow_reasoning_tag: bool) -> str:
    s = text.strip()
    # code fences
    m = re.match(r"(?s)^```(?:json|xml)?\s*(.*?)\s*```$", s)
    if m:
        s = m.group(1).strip()
    # leading <think>...</think>
    if allow_reasoning_tag:
        s = re.sub(r"(?s)^<think>.*?</think>\s*", "", s)
    return s.strip()

def parse_schedule_any(text: str, allow_reasoning_tag: bool) -> Optional[List[Dict[str,str]]]:
    s = strip_fences_and_maybe_think(text, allow_reasoning_tag)

    # Try JSON: {"schedule":[{"name":..., "start":"HH:MM","end":"HH:MM"}, ...]}
    try:
        obj = json.loads(s)
        if isinstance(obj, dict) and isinstance(obj.get("schedule"), list):
            out = []
            for e in obj["schedule"]:
                if not all(k in e for k in ("name","start","end")):
                    return None
                out.append({"name": str(e["name"]), "start": str(e["start"]), "end": str(e["end"])})
            return out
    except Exception:
        pass

    # Try XML:
    # <schedule><event><name>..</name><start>HH:MM</start><end>HH:MM</end></event>...</schedule>
    try:
        root = ET.fromstring(s)
        if root.tag.lower() == "schedule":
            out = []
            for ev in root.findall(".//event"):
                name = (ev.findtext("name") or "").strip()
                start = (ev.findtext("start") or "").strip()
                end = (ev.findtext("end") or "").strip()
                if not (name and start and end):
                    return None
                out.append({"name": name, "start": start, "end": end})
            if out:
                return out
    except Exception:
        pass

    return None
