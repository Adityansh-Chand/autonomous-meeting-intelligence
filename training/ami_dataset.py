"""Turn the AMI annotations into a labelled decision / action-item dataset.

AMI does not ship this dataset; it ships the parts. Three layers have to be
joined:

    words/          the transcript, one XML element per token
    dialogueActs/   utterance boundaries, each pointing at a span of word ids
    abstractive/    human summaries, grouped under ABSTRACT / DECISIONS /
                    ACTIONS / PROBLEMS headings
    extractive/     summlink: which dialogue act supports which summary sentence

A dialogue act linked to a sentence under DECISIONS is a decision; one linked
under ACTIONS is an action item; everything else is `neither`. That is the
construction the published work on this task uses, and it is why AMI can serve as
ground truth here at all: the labels come from human annotators summarising a
meeting, not from anyone looking for a pattern a classifier could find.
"""
import re
import zipfile
import xml.etree.ElementTree as ET
from collections import defaultdict

NITE = "http://nite.sourceforge.net/"
NS = {"nite": NITE}
SPEAKERS = "ABCDE"

# Section tag in the abstractive summary -> our label.
SECTION_LABELS = {"decisions": "decision", "actions": "action_item"}
NEITHER = "neither"


def _id(element):
    return element.get(f"{{{NITE}}}id")


def _href_ids(href):
    return re.findall(r"id\(([^)]+)\)", href or "")


def meetings_in(archive):
    """Meeting ids that carry an abstractive summary. The rest cannot be labelled."""
    return sorted({
        name.split("/")[1].split(".")[0]
        for name in archive.namelist()
        if name.startswith("abstractive/") and name.endswith(".xml")
    })


def _words(archive, names, meeting, speaker):
    path = f"words/{meeting}.{speaker}.words.xml"
    if path not in names:
        return {}, []
    root = ET.fromstring(archive.read(path))
    text, order = {}, []
    for element in root:
        word_id = _id(element)
        if word_id is None:
            continue
        order.append(word_id)
        # Non-<w> elements are vocal sounds, gaps and the like: they hold a
        # position in the sequence but contribute no text.
        text[word_id] = element.text or "" if element.tag == "w" else ""
    return text, order


def _utterances(archive, names, meeting, speaker):
    """dialogue-act id -> utterance text, reconstructed from its word span."""
    path = f"dialogueActs/{meeting}.{speaker}.dialog-act.xml"
    if path not in names:
        return {}
    text, order = _words(archive, names, meeting, speaker)
    position = {word_id: index for index, word_id in enumerate(order)}

    root = ET.fromstring(archive.read(path))
    utterances = {}
    for act in root.findall("dact"):
        child = act.find("nite:child", NS)
        if child is None:
            continue
        ids = _href_ids(child.get("href"))
        if not ids:
            continue
        if len(ids) == 1:
            span = [ids[0]]
        else:
            start, end = position.get(ids[0]), position.get(ids[1])
            span = order[start:end + 1] if start is not None and end is not None else []
        joined = " ".join(text.get(word_id, "") for word_id in span)
        joined = re.sub(r"\s+([.,?!])", r"\1", re.sub(r"\s+", " ", joined)).strip()
        if joined:
            utterances[_id(act)] = joined
    return utterances


def _summary_sections(archive, meeting):
    """abstractive sentence id -> section tag (abstract/decisions/actions/...)."""
    root = ET.fromstring(archive.read(f"abstractive/{meeting}.abssumm.xml"))
    sections = {}
    for section in root:
        for sentence in section.findall("sentence"):
            sections[_id(sentence)] = section.tag
    return sections


def _links(archive, names, meeting, sections):
    """dialogue-act id -> set of summary section tags it supports."""
    path = f"extractive/{meeting}.summlink.xml"
    if path not in names:
        return {}
    root = ET.fromstring(archive.read(path))
    linked = defaultdict(set)
    for link in root.findall("summlink"):
        extractive = abstractive = None
        for pointer in link.findall("nite:pointer", NS):
            ids = _href_ids(pointer.get("href"))
            if not ids:
                continue
            if pointer.get("role") == "extractive":
                extractive = ids[0]
            else:
                abstractive = ids[0]
        if extractive and abstractive and abstractive in sections:
            linked[extractive].add(sections[abstractive])
    return linked


def build(archive_path):
    """-> (examples, stats). Each example is (meeting, text, label)."""
    examples, ambiguous, unlinked_meetings = [], 0, 0
    with zipfile.ZipFile(archive_path) as archive:
        names = set(archive.namelist())
        for meeting in meetings_in(archive):
            sections = _summary_sections(archive, meeting)
            linked = _links(archive, names, meeting, sections)
            if not linked:
                # No summlink file: the meeting is summarised but not aligned, so
                # every utterance would be labelled `neither` by default. Dropping
                # it is right -- keeping it would add thousands of false negatives.
                unlinked_meetings += 1
                continue
            for speaker in SPEAKERS:
                for act_id, text in _utterances(archive, names, meeting, speaker).items():
                    tags = linked.get(act_id, set())
                    labels = {SECTION_LABELS[t] for t in tags if t in SECTION_LABELS}
                    if len(labels) > 1:
                        # Annotators linked this utterance to both a decision and
                        # an action. Real, but not representable in a 3-class
                        # scheme, so it is dropped and counted rather than
                        # silently assigned to whichever class we prefer.
                        ambiguous += 1
                        continue
                    label = labels.pop() if labels else NEITHER
                    examples.append((meeting, text, label))
    stats = {
        "ambiguous_dropped": ambiguous,
        "meetings_without_summlink": unlinked_meetings,
    }
    return examples, stats
