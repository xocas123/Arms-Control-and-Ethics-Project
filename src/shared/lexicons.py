"""
Shared lexicons for humanitarian vs deterrence frame classification.
Used by all four research question modules.
"""
import re

HUMANITARIAN = [
    "humanitarian", "civilian", "suffering", "indiscriminate",
    "disproportionate", "unnecessary suffering", "human cost",
    "humanitarian consequence", "humanitarian impact", "victim",
    "civilian protection", "civilian casualty", "human rights",
    "international humanitarian law", "distinction", "proportionality",
    "inhumane", "unacceptable harm", "catastrophic", "unconscionable",
    "humanitarian disarmament", "human security", "human dignity",
    "civilian harm", "civilian deaths", "civilian population",
    "explosive remnant", "victim assistance", "mine clearance",
    "contamination", "explosive ordnance", "prohibited weapon",
    "indiscriminate effect", "superfluous injury", "unnecessary harm",
]

DETERRENCE = [
    "deterrence", "deterrent", "strategic stability", "balance",
    "security guarantee", "nuclear umbrella", "extended deterrence",
    "mutual assured", "credible threat", "first strike", "second strike",
    "strategic parity", "arms race stability", "escalation control",
    "national security", "vital interest", "defense posture",
    "military balance", "force structure", "readiness",
    "nuclear doctrine", "strategic forces", "credible minimum deterrent",
    "second-strike", "first-strike", "launch-on-warning",
    "strategic reserve", "nuclear triad", "delivery system",
    "modernization", "recapitalization", "force modernization",
]

# Concepts for concept-level sentiment analysis (Q4)
CONCEPTS = [
    "nuclear disarmament",
    "non-proliferation",
    "deterrence",
    "ban treaty",
    "prohibition",
    "TPNW",
    "arms trade",
    "humanitarian consequence",
    "verification",
    "compliance",
]

# Treaty-specific lexicons for Q3 cross-validation
TREATY_LEXICONS = {
    "ottawa": [
        "humanitarian", "civilian", "indiscriminate", "landmine",
        "anti-personnel", "mine clearance", "victim assistance",
        "explosive remnant", "mine-free",
    ],
    "ccm": [
        "humanitarian", "civilian", "indiscriminate", "cluster munition",
        "submunition", "explosive remnant", "dud rate",
    ],
    "att": [
        "responsible transfer", "risk assessment", "human rights",
        "diversion", "end-use", "end-user", "arms transfer",
        "brokering", "transit", "transhipment",
    ],
    "tpnw": [
        "prohibition", "catastrophic", "humanitarian consequence",
        "ban", "nuclear weapon", "stigma", "stigmatize",
        "hibakusha", "survivors", "testing victims",
    ],
    "laws_drones": [
        "lethal autonomous weapon", "LAWS", "meaningful human control",
        "autonomous weapon", "killer robot", "human-in-the-loop",
        "CCW", "Convention on Certain Conventional Weapons",
        "Group of Governmental Experts", "GGE",
        "drone", "unmanned aerial", "remotely piloted",
        "armed drone", "drone transfer",
    ],
}

DRONE_AUTONOMOUS = [
    "drone", "drones", "unmanned aerial vehicle", "UAV",
    "unmanned combat aerial vehicle", "UCAV", "UAS",
    "remotely piloted aircraft", "remotely piloted",
    "autonomous weapon", "lethal autonomous weapon", "LAWS",
    "killer robot", "meaningful human control",
    "human-in-the-loop", "human-on-the-loop", "human oversight",
    "autonomous system", "autonomous military",
    "armed drone", "drone strike", "targeted killing",
    "drone proliferation", "drone transfer",
    "loitering munition", "one-way attack",
    "algorithmic warfare", "AI weapon",
    "accountability gap",
]


def count_matches(text: str, lexicon: list) -> int:
    """Count phrase/word matches from lexicon in text (case-insensitive)."""
    if not text:
        return 0
    text_lower = text.lower()
    count = 0
    for phrase in lexicon:
        phrase_lower = phrase.lower()
        # Whole-word matching for single words, substring for phrases
        if " " in phrase_lower:
            count += text_lower.count(phrase_lower)
        else:
            count += len(re.findall(r'\b' + re.escape(phrase_lower) + r'\b', text_lower))
    return count


def compute_frame_ratio_lexicon(text: str) -> float:
    """
    Compute humanitarian frame ratio using lexicon counts.
    Returns h_count / max(h_count + d_count, 1).
    0 = pure deterrence, 1 = pure humanitarian.
    """
    h_count = count_matches(text, HUMANITARIAN)
    d_count = count_matches(text, DETERRENCE)
    return h_count / max(h_count + d_count, 1)


def count_treaty_lexicon(text: str, treaty: str) -> int:
    """Count matches for a specific treaty's lexicon."""
    lexicon = TREATY_LEXICONS.get(treaty, [])
    return count_matches(text, lexicon)
