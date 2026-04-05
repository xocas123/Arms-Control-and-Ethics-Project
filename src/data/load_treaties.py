"""Load treaty anchor passages from JSON file."""
import json
import os
from pathlib import Path

def load_treaty_anchors(anchors_file="data/treaties/anchors.json"):
    """Load treaty anchor passages. Returns dict of treaty data."""
    path = Path(anchors_file)
    if not path.exists():
        print(f"[Treaties] Anchors file not found at {anchors_file}. Using hardcoded fallback.")
        return _hardcoded_anchors()

    with open(path, "r", encoding="utf-8") as f:
        anchors = json.load(f)
    print(f"[Treaties] Loaded anchors for: {', '.join(anchors.keys())}")
    return anchors


def get_anchor_texts(anchors, treaty, passage_names=None):
    """Returns list of passage texts for a treaty."""
    if treaty not in anchors:
        return []
    passages = anchors[treaty].get("passages", {})
    if passage_names:
        return [passages[p] for p in passage_names if p in passages]
    return list(passages.values())


def get_humanitarian_anchor_texts(anchors):
    """Texts from TPNW preamble, Ottawa preamble, CCM preamble, ATT preamble."""
    texts = []
    for treaty, passage in [("tpnw", "preamble"), ("ottawa", "preamble"),
                              ("ccm", "preamble"), ("att", "preamble")]:
        t = anchors.get(treaty, {}).get("passages", {}).get(passage)
        if t:
            texts.append(t)
    return texts


def get_security_anchor_texts(anchors):
    """Texts from NPT article VI + CWC preamble."""
    texts = []
    for treaty, passage in [("npt", "article_vi"), ("npt", "preamble"), ("cwc", "preamble")]:
        t = anchors.get(treaty, {}).get("passages", {}).get(passage)
        if t:
            texts.append(t)
    return texts


def _hardcoded_anchors():
    """Fallback anchors if JSON file not found."""
    return {
        "tpnw": {
            "frame": "humanitarian prohibition of nuclear weapons",
            "year": 2017,
            "passages": {
                "preamble": "Deeply concerned about the catastrophic humanitarian consequences that would result from any use of nuclear weapons, and recognizing the consequent need to completely eliminate such weapons, which remains the only way to guarantee that nuclear weapons are never used again under any circumstances. Mindful of the unacceptable suffering of and harm caused to the victims of the use of nuclear weapons (hibakusha), as well as of those affected by the testing of nuclear weapons.",
                "article_1": "Each State Party undertakes never under any circumstances to develop, test, produce, manufacture, otherwise acquire, possess or stockpile nuclear weapons or other nuclear explosive devices; transfer to any recipient whatsoever nuclear weapons or other nuclear explosive devices or control over such weapons or explosive devices directly or indirectly; use or threaten to use nuclear weapons or other nuclear explosive devices."
            }
        },
        "npt": {
            "frame": "non-proliferation / disarmament obligation",
            "year": 1968,
            "passages": {
                "preamble": "Declaring their intention to achieve at the earliest possible date the cessation of the nuclear arms race and to undertake effective measures in the direction of nuclear disarmament, and urging the cooperation of all States in the attainment of this objective.",
                "article_vi": "Each of the Parties to the Treaty undertakes to pursue negotiations in good faith on effective measures relating to cessation of the nuclear arms race at an early date and to nuclear disarmament, and on a treaty on general and complete disarmament under strict and effective international control."
            }
        },
        "ottawa": {
            "frame": "humanitarian ban",
            "year": 1997,
            "passages": {
                "preamble": "Determined to put an end for all time to the suffering and casualties caused by anti-personnel mines, that kill or maim hundreds of people every week, mostly innocent and defenceless civilians and especially children, obstruct economic development and reconstruction, inhibit the repatriation of refugees and internally displaced persons, and have other severe consequences for years after emplacement."
            }
        },
        "att": {
            "frame": "responsible transfer / human rights risk assessment",
            "year": 2013,
            "passages": {
                "preamble": "Recognizing the legitimate political, security, economic and commercial interests of States in the international trade in conventional arms. Reaffirming the right of all States to manufacture, acquire, export, import and transfer conventional arms. Mindful of the important role that civil society organizations and industry play in raising awareness of and contributing to the prevention of the illicit trade in conventional arms.",
                "article_6": "A State Party shall not authorize any transfer of conventional arms covered under Article 2(1) or of items covered under Article 3 or Article 4, if it has knowledge at the time of authorization that the arms or items would be used in the commission of genocide, crimes against humanity, grave breaches of the Geneva Conventions of 1949, attacks directed against civilian objects or civilians protected as such, or other war crimes.",
                "article_7": "If the export is not prohibited under Article 6, each exporting State Party, prior to authorization of export of conventional arms covered under Article 2(1), shall assess the potential that the conventional arms to be exported would contribute to or undermine peace and security, and whether they could be used to commit or facilitate a serious violation of international humanitarian law."
            }
        }
    }
