"""
Synthetic UNGDC corpus generator.

Generates realistic full-length General Debate speeches for ~80 countries,
years 2000-2023.  Each country-year gets ONE full speech (600-1,500 words)
covering multiple topics — arms control content is interspersed with
development, trade, and general security paragraphs, mirroring the real corpus.

The segmentation step (segment.py) is then responsible for extracting
arms-control-relevant paragraphs — just like on the real UNGDC corpus.

Usage:
    python -m src.data.synthetic_ungdc
    from src.data.synthetic_ungdc import generate_synthetic_corpus
"""

import random
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Treaty anchor passages — injected verbatim to give real normative signal
# ---------------------------------------------------------------------------

_TREATY_ANCHORS = {
    "npt": (
        "Each of the Parties to the Treaty undertakes to pursue negotiations in good faith "
        "on effective measures relating to cessation of the nuclear arms race at an early date "
        "and to nuclear disarmament, and on a treaty on general and complete disarmament under "
        "strict and effective international control."
    ),
    "att": (
        "If the export is not prohibited under Article 6, each exporting State Party, prior to "
        "authorization of the export of conventional arms, shall assess the potential that the "
        "conventional arms would contribute to or undermine peace and security or could be used "
        "to commit or facilitate a serious violation of international humanitarian law."
    ),
    "tpnw": (
        "Deeply concerned about the catastrophic humanitarian consequences that would result "
        "from any use of nuclear weapons, and recognizing the consequent need to completely "
        "eliminate such weapons. Mindful of the unacceptable suffering of and harm caused to "
        "the victims of the use of nuclear weapons, the hibakusha."
    ),
    "ottawa": (
        "Determined to put an end for all time to the suffering and casualties caused by "
        "anti-personnel mines, that kill or maim hundreds of people every week, many of them "
        "children, obstruct economic development and reconstruction, inhibit the repatriation "
        "of refugees and internally displaced persons."
    ),
    "cwc": (
        "Each State Party undertakes never under any circumstances to develop, produce, "
        "stockpile or otherwise acquire or retain chemical weapons. The verification regime "
        "of the Chemical Weapons Convention under the OPCW represents the gold standard "
        "for arms control accountability and compliance."
    ),
    "ccm": (
        "Recognizing the serious violations of international humanitarian law caused by the "
        "indiscriminate area effects and the dangers of unexploded submunitions posed by cluster "
        "munitions, which kill or maim civilians, including women and children, both during "
        "attacks and afterwards."
    ),
    "bwc": (
        "Convinced that such use of bacteriological and toxin weapons would be repugnant to "
        "the conscience of mankind and that no effort should be spared to minimize this risk. "
        "The development, production and stockpiling of biological weapons constitutes a "
        "violation of the principles of humanity."
    ),
}

# ---------------------------------------------------------------------------
# Phrase pools — arms control content by rhetorical register
# ---------------------------------------------------------------------------

_NUCLEAR_DISARMAMENT = [
    "nuclear disarmament remains the highest priority of the international community",
    "we call upon all nuclear weapon states to fulfill their obligations under Article VI of the NPT",
    "the Non-Proliferation Treaty is the cornerstone of the global non-proliferation regime",
    "we urge all states to ratify the Comprehensive Nuclear-Test-Ban Treaty without further delay",
    "the Conference on Disarmament must resume substantive work without further delay",
    "multilateral negotiations on nuclear disarmament must be pursued in good faith",
    "we support the establishment of nuclear-weapon-free zones in all regions of the world",
    "the IAEA safeguards system must be strengthened and universalized",
    "nuclear weapons pose an existential threat to all of humanity and must be eliminated",
    "we reaffirm the need for a world free of nuclear weapons as a global public good",
    "a fissile material cut-off treaty must be negotiated without further preconditions",
    "the NPT review process must yield concrete, time-bound progress on disarmament commitments",
    "we are alarmed by the modernization of nuclear arsenals and the development of new delivery systems",
    "the risk of nuclear weapons use — whether by miscalculation, accident, or design — is unacceptably high",
    "we call for entry into force of the CTBT as an immediate and essential disarmament measure",
    "negative security assurances must be legally binding and unconditional",
    "unilateral measures toward nuclear disarmament build confidence and momentum for multilateral progress",
    "we support the establishment of a Middle East zone free of weapons of mass destruction",
    "the nuclear disarmament obligations of the NPT nuclear weapon states remain unfulfilled",
    "strategic stability cannot be a permanent substitute for actual disarmament",
]

_HUMANITARIAN = [
    "the humanitarian consequences of nuclear weapons use would be catastrophic and irreversible",
    "civilian protection must be at the center of all arms control discussions",
    "indiscriminate weapons cause unacceptable harm to non-combatants and civilian populations",
    "the Treaty on the Prohibition of Nuclear Weapons represents a historic humanitarian achievement",
    "we call on all states to join the TPNW without conditions or delay",
    "the suffering of hibakusha must never be forgotten — their testimony demands action",
    "principles of distinction and proportionality under international humanitarian law must govern all weapons use",
    "anti-personnel mines and cluster munitions continue to kill civilians decades after conflict ends",
    "explosive remnants of war pose long-term, indiscriminate dangers to post-conflict communities",
    "the indiscriminate area effects of cluster munitions are incompatible with humanitarian law",
    "victim assistance and mine clearance must be integral to all disarmament frameworks",
    "we champion the voices of civil society and affected communities in disarmament negotiations",
    "the humanitarian imperative demands that we move beyond deterrence thinking",
    "no state's security interest can justify the deliberate targeting of civilian populations",
    "autonomous weapons systems must be prohibited before they are deployed at scale",
    "lethal autonomous weapons raise fundamental questions about accountability under IHL",
    "we support a political declaration on explosive weapons in populated areas",
    "gender perspectives must be integrated into disarmament policy and implementation",
    "the disproportionate impact of weapons on women and children demands urgent attention",
    "we stand in solidarity with all communities devastated by the humanitarian consequences of armed violence",
]

_DETERRENCE = [
    "nuclear deterrence continues to play an essential role in ensuring strategic stability",
    "our security commitments to allies depend on maintaining credible extended deterrent capabilities",
    "we support responsible arms control that preserves strategic stability and international security",
    "missile defense systems contribute to deterrence by reducing the incentive for first strikes",
    "extended deterrence arrangements remain vital for the security of our allies and partners",
    "we are committed to arms control that is verifiable, balanced, and enhances security for all",
    "disarmament must proceed in a manner consistent with the principle of undiminished security",
    "we oppose measures that would undermine the deterrent posture of responsible states",
    "strategic stability requires meaningful transparency and risk reduction measures",
    "we support further reductions in nuclear arsenals through bilateral and multilateral frameworks",
    "confidence-building measures reduce the risk of miscalculation and accidental escalation",
    "arms control must be accompanied by adequate verification to be credible",
    "we maintain a defensive nuclear posture and do not target any state with our strategic forces",
    "responsible stewardship of nuclear weapons requires constant attention to safety and security",
    "we support the modernization of nuclear command and control systems to reduce accidental launch risks",
    "the nuclear triad provides resilience against first-strike scenarios and enhances crisis stability",
]

_SOVEREIGNTY = [
    "we reject the selective application of arms control norms that target specific states",
    "sovereignty principles must be respected in all international security arrangements",
    "double standards in arms control implementation undermine the legitimacy of the international order",
    "unilateral sanctions disguised as non-proliferation measures are counterproductive and illegitimate",
    "the security concerns of all states must be addressed in a balanced and equitable manner",
    "we oppose politically motivated arms embargoes that destabilize regional security",
    "the international community must address root causes of insecurity rather than symptoms",
    "arms control cannot be imposed but must be negotiated on an equal and sovereign footing",
    "we call for a democratic and multipolar international security architecture",
    "regional security concerns must be taken into account in any global disarmament agenda",
    "every state has the inherent right of self-defense under Article 51 of the UN Charter",
    "selective verification regimes applied only to non-Western states undermine treaty legitimacy",
    "we will not accept discriminatory arrangements that codify the privileges of the powerful",
    "the use of arms control as a geopolitical tool is incompatible with multilateral principles",
    "interference in states' legitimate defense programs violates international law",
]

_CONVENTIONAL_ARMS = [
    "the Arms Trade Treaty establishes critical norms for responsible conventional arms transfers",
    "we call on major arms exporters to apply stringent human rights criteria in transfer decisions",
    "illicit arms trafficking fuels conflict, terrorism, and organized crime across multiple regions",
    "small arms and light weapons proliferation remains a critical and persistent security challenge",
    "we support the Programme of Action to prevent, combat and eradicate illicit small arms traffic",
    "transparency in arms transfers promotes accountability and reduces destabilizing accumulations",
    "the ATT must be universalized and its implementation strengthened through national measures",
    "arms brokers and intermediaries must be regulated under robust national export control legislation",
    "end-user monitoring and post-shipment verification are essential safeguards against diversion",
    "we urge all states to submit annual reports under the UN Register of Conventional Arms",
    "we are deeply concerned about the transfer of advanced weapons systems to conflict zones",
    "arms transfers to human rights abusers undermine the credibility of the exporting state",
    "the proliferation of man-portable air defense systems poses a growing threat to civil aviation",
    "national stockpile management and weapons marking are essential to prevent leakage to illicit markets",
    "we support regional arms control agreements as complements to global treaty frameworks",
]

_REGIONAL_SECURITY = [
    "regional security dynamics must be addressed as part of any comprehensive disarmament agenda",
    "the proliferation of advanced ballistic missile systems threatens regional and global stability",
    "we are deeply concerned about the escalating militarization of ongoing conflict zones",
    "proxy conflicts fueled by external arms transfers undermine diplomatic resolution efforts",
    "a comprehensive arms embargo is necessary to prevent further atrocities and civilian casualties",
    "we call for an immediate cessation of hostilities and unimpeded humanitarian access",
    "the security of our region cannot be separated from global arms control norms and commitments",
    "counterterrorism efforts must be coordinated with responsible arms transfer control policies",
    "we support confidence-building measures among regional states to reduce bilateral tensions",
    "maritime security cooperation is essential to prevent arms smuggling by non-state actors",
    "we are alarmed by the transfer of weapons to non-state armed groups and terrorist organizations",
    "regional arms races driven by great power competition must be addressed at the multilateral level",
]

_VERIFICATION = [
    "robust verification and compliance mechanisms are the foundation of credible arms control",
    "on-site inspections must be permitted without obstruction, delay, or politicization",
    "data exchanges and transparency measures build confidence among treaty parties over time",
    "the OPCW has demonstrated that rigorous verification of chemical weapons elimination is achievable",
    "we call for a strengthened IAEA mandate with universal additional protocol implementation",
    "non-compliance with treaty obligations must carry meaningful and consistent consequences",
    "trust but verify remains the appropriate and necessary framework for arms control implementation",
    "national implementation of treaty obligations requires both legislative frameworks and technical capacity",
    "the UN Secretary-General's mechanism for investigating alleged chemical weapons use must be preserved",
    "multilateral verification bodies must be adequately funded and protected from political interference",
    "we support the development of new verification technologies for nuclear disarmament",
    "any future disarmament arrangement will require unprecedented levels of transparency and monitoring",
]

_CYBER_EMERGING = [
    "emerging technologies including autonomous weapons systems require urgent normative attention",
    "lethal autonomous weapons systems must remain subject to meaningful human control",
    "we call for a legally binding instrument to prohibit fully autonomous weapons systems",
    "cyberattacks on critical infrastructure may constitute violations of international humanitarian law",
    "the militarization of artificial intelligence poses new, unpredictable, and existential risks",
    "dual-use technology transfer controls must adapt urgently to emerging threat environments",
    "we support the development of confidence-building measures in cyberspace to reduce escalation risks",
    "outer space must remain free from the placement of weapons and destabilizing arms race dynamics",
    "hypersonic weapons and advanced delivery systems threaten strategic stability and warning time",
    "we call for a moratorium on antisatellite weapon tests that generate persistent debris fields",
    "algorithmic decision-making in weapons systems must be subject to international legal scrutiny",
    "we support a group of governmental experts process on lethal autonomous weapons systems",
]

# Non-arms-control filler paragraphs — the real UNGDC speeches cover many topics
_DEVELOPMENT_FILLER = [
    "sustainable development and climate action are inseparable from international peace and security",
    "we call for the full implementation of the 2030 Agenda and the Sustainable Development Goals",
    "climate change represents an existential threat requiring urgent multilateral action",
    "the COVID-19 pandemic has exposed deep inequalities in the global health architecture",
    "we call for equitable access to vaccines and medical countermeasures for all developing states",
    "debt relief and concessional financing are urgently needed for least developed countries",
    "food security cannot be achieved without addressing the root causes of conflict and displacement",
    "the digital divide must be addressed to ensure no country is left behind in the digital economy",
    "we reaffirm our commitment to official development assistance targets and South-South cooperation",
    "trade liberalization and market access remain essential for developing country economic growth",
    "migration and displacement require comprehensive and rights-based multilateral responses",
    "we call for reform of international financial institutions to better represent developing countries",
]

_HUMAN_RIGHTS_FILLER = [
    "the promotion and protection of human rights is a fundamental pillar of the United Nations",
    "we condemn violations of international humanitarian law wherever they occur",
    "the rule of law and independent judicial institutions are the foundation of durable peace",
    "freedom of expression, assembly, and the press must be protected in all circumstances",
    "we call for an end to impunity for war crimes, crimes against humanity, and genocide",
    "transitional justice and accountability mechanisms are essential for durable reconciliation",
    "the rights of women, children, and minorities must be at the center of our peace efforts",
    "we support the work of the International Criminal Court as a cornerstone of global justice",
]

_GENERAL_UN_FILLER = [
    "reform of the United Nations Security Council is long overdue and must reflect 21st century realities",
    "multilateralism remains the only viable framework for addressing global challenges",
    "we reaffirm our commitment to the principles enshrined in the Charter of the United Nations",
    "the revitalization of the General Assembly must be pursued with urgency and determination",
    "we call for a more equitable and representative international order",
    "dialogue and diplomacy must always be preferred over confrontation and coercion",
    "we support peacekeeping operations and special political missions in conflict-affected regions",
    "the United Nations must be equipped with adequate resources to fulfill its mandated responsibilities",
]

# ---------------------------------------------------------------------------
# Country profiles
# ---------------------------------------------------------------------------

COUNTRY_LIST = [
    # P5 / major exporters
    ("USA", "United States"), ("RUS", "Russia"), ("CHN", "China"),
    ("GBR", "United Kingdom"), ("FRA", "France"), ("DEU", "Germany"),
    # De facto nuclear / NPT outliers
    ("IND", "India"), ("PAK", "Pakistan"), ("ISR", "Israel"), ("PRK", "North Korea"),
    # Humanitarian coalition / TPNW champions
    ("AUT", "Austria"), ("IRL", "Ireland"), ("NZL", "New Zealand"),
    ("MEX", "Mexico"), ("BRA", "Brazil"), ("ZAF", "South Africa"),
    ("CRI", "Costa Rica"), ("BOL", "Bolivia"), ("ECU", "Ecuador"),
    # NATO / EU allies
    ("TUR", "Turkey"), ("POL", "Poland"), ("ITA", "Italy"), ("ESP", "Spain"),
    ("NLD", "Netherlands"), ("BEL", "Belgium"), ("NOR", "Norway"), ("DNK", "Denmark"),
    ("CAN", "Canada"), ("AUS", "Australia"), ("JPN", "Japan"), ("KOR", "South Korea"),
    ("SWE", "Sweden"), ("FIN", "Finland"), ("GRC", "Greece"), ("HUN", "Hungary"),
    ("CZE", "Czech Republic"), ("PRT", "Portugal"), ("CHE", "Switzerland"),
    # Gulf states
    ("SAU", "Saudi Arabia"), ("ARE", "United Arab Emirates"),
    ("QAT", "Qatar"), ("KWT", "Kuwait"), ("BHR", "Bahrain"), ("OMN", "Oman"),
    # Middle East / conflict-adjacent
    ("IRN", "Iran"), ("IRQ", "Iraq"), ("SYR", "Syria"), ("JOR", "Jordan"),
    ("LBN", "Lebanon"), ("YEM", "Yemen"), ("LBY", "Libya"),
    # South / Southeast Asia
    ("THA", "Thailand"), ("IDN", "Indonesia"), ("MYS", "Malaysia"),
    ("SGP", "Singapore"), ("PHL", "Philippines"), ("VNM", "Vietnam"),
    ("BGD", "Bangladesh"), ("LKA", "Sri Lanka"), ("MMR", "Myanmar"),
    ("KAZ", "Kazakhstan"), ("UZB", "Uzbekistan"),
    # Africa
    ("NGA", "Nigeria"), ("ETH", "Ethiopia"), ("EGY", "Egypt"),
    ("ZAF", "South Africa"), ("KEN", "Kenya"), ("TZA", "Tanzania"),
    ("GHA", "Ghana"), ("DZA", "Algeria"), ("MAR", "Morocco"),
    ("SDN", "Sudan"), ("SOM", "Somalia"), ("MLI", "Mali"),
    # Americas
    ("COL", "Colombia"), ("ARG", "Argentina"), ("CHL", "Chile"),
    ("PER", "Peru"), ("URY", "Uruguay"), ("VEN", "Venezuela"),
    ("CUB", "Cuba"),
    # Europe
    ("UKR", "Ukraine"), ("SRB", "Serbia"), ("BLR", "Belarus"),
    ("AZE", "Azerbaijan"), ("GEO", "Georgia"),
    # Other
    ("NAM", "Namibia"), ("SEN", "Senegal"),
]
# De-duplicate
_seen = set()
COUNTRY_LIST = [(iso, name) for iso, name in COUNTRY_LIST
                if iso not in _seen and not _seen.add(iso)]

_COUNTRY_PROFILES = {
    "USA": {
        "arms_pools": [
            (_DETERRENCE, 0.35), (_NUCLEAR_DISARMAMENT, 0.25),
            (_CONVENTIONAL_ARMS, 0.20), (_VERIFICATION, 0.15), (_CYBER_EMERGING, 0.05),
        ],
        "anchors": ["npt", "att"], "anchor_prob": 0.20,
        "arms_paragraphs": (3, 5),
    },
    "GBR": {
        "arms_pools": [
            (_DETERRENCE, 0.30), (_NUCLEAR_DISARMAMENT, 0.25),
            (_CONVENTIONAL_ARMS, 0.25), (_VERIFICATION, 0.15), (_HUMANITARIAN, 0.05),
        ],
        "anchors": ["npt", "att"], "anchor_prob": 0.20,
        "arms_paragraphs": (3, 5),
    },
    "FRA": {
        "arms_pools": [
            (_DETERRENCE, 0.30), (_NUCLEAR_DISARMAMENT, 0.25),
            (_CONVENTIONAL_ARMS, 0.25), (_VERIFICATION, 0.15), (_HUMANITARIAN, 0.05),
        ],
        "anchors": ["npt", "att"], "anchor_prob": 0.20,
        "arms_paragraphs": (3, 5),
    },
    "DEU": {
        "arms_pools": [
            (_CONVENTIONAL_ARMS, 0.30), (_NUCLEAR_DISARMAMENT, 0.25),
            (_HUMANITARIAN, 0.20), (_VERIFICATION, 0.15), (_DETERRENCE, 0.10),
        ],
        "anchors": ["att", "npt"], "anchor_prob": 0.28,
        "arms_paragraphs": (3, 5),
    },
    "AUT": {
        "arms_pools": [
            (_HUMANITARIAN, 0.50), (_NUCLEAR_DISARMAMENT, 0.30),
            (_VERIFICATION, 0.12), (_CONVENTIONAL_ARMS, 0.08),
        ],
        "anchors": ["tpnw", "ottawa", "ccm"], "anchor_prob": 0.45,
        "arms_paragraphs": (4, 7),
    },
    "IRL": {
        "arms_pools": [
            (_HUMANITARIAN, 0.48), (_NUCLEAR_DISARMAMENT, 0.30),
            (_VERIFICATION, 0.12), (_CONVENTIONAL_ARMS, 0.10),
        ],
        "anchors": ["tpnw", "npt"], "anchor_prob": 0.42,
        "arms_paragraphs": (4, 7),
    },
    "NZL": {
        "arms_pools": [
            (_HUMANITARIAN, 0.45), (_NUCLEAR_DISARMAMENT, 0.35),
            (_CONVENTIONAL_ARMS, 0.12), (_VERIFICATION, 0.08),
        ],
        "anchors": ["tpnw", "npt", "ccm"], "anchor_prob": 0.42,
        "arms_paragraphs": (4, 6),
    },
    "MEX": {
        "arms_pools": [
            (_HUMANITARIAN, 0.42), (_NUCLEAR_DISARMAMENT, 0.35),
            (_CONVENTIONAL_ARMS, 0.13), (_REGIONAL_SECURITY, 0.10),
        ],
        "anchors": ["tpnw", "att"], "anchor_prob": 0.40,
        "arms_paragraphs": (3, 6),
    },
    "ZAF": {
        "arms_pools": [
            (_HUMANITARIAN, 0.35), (_NUCLEAR_DISARMAMENT, 0.35),
            (_CONVENTIONAL_ARMS, 0.18), (_REGIONAL_SECURITY, 0.12),
        ],
        "anchors": ["tpnw", "npt"], "anchor_prob": 0.38,
        "arms_paragraphs": (3, 5),
    },
    "BRA": {
        "arms_pools": [
            (_HUMANITARIAN, 0.35), (_NUCLEAR_DISARMAMENT, 0.30),
            (_CONVENTIONAL_ARMS, 0.20), (_REGIONAL_SECURITY, 0.15),
        ],
        "anchors": ["tpnw", "npt"], "anchor_prob": 0.35,
        "arms_paragraphs": (3, 5),
    },
    "RUS": {
        "arms_pools": [
            (_SOVEREIGNTY, 0.35), (_NUCLEAR_DISARMAMENT, 0.25),
            (_DETERRENCE, 0.25), (_VERIFICATION, 0.10), (_CYBER_EMERGING, 0.05),
        ],
        "anchors": ["npt"], "anchor_prob": 0.18,
        "arms_paragraphs": (3, 5),
    },
    "CHN": {
        "arms_pools": [
            (_SOVEREIGNTY, 0.35), (_NUCLEAR_DISARMAMENT, 0.30),
            (_DETERRENCE, 0.20), (_REGIONAL_SECURITY, 0.10), (_VERIFICATION, 0.05),
        ],
        "anchors": ["npt"], "anchor_prob": 0.18,
        "arms_paragraphs": (3, 5),
    },
    "IND": {
        "arms_pools": [
            (_NUCLEAR_DISARMAMENT, 0.30), (_SOVEREIGNTY, 0.25),
            (_DETERRENCE, 0.20), (_REGIONAL_SECURITY, 0.15), (_VERIFICATION, 0.10),
        ],
        "anchors": ["npt"], "anchor_prob": 0.15,
        "arms_paragraphs": (3, 5),
    },
    "PAK": {
        "arms_pools": [
            (_NUCLEAR_DISARMAMENT, 0.28), (_SOVEREIGNTY, 0.27),
            (_REGIONAL_SECURITY, 0.25), (_DETERRENCE, 0.15), (_VERIFICATION, 0.05),
        ],
        "anchors": ["npt"], "anchor_prob": 0.15,
        "arms_paragraphs": (3, 5),
    },
    "SAU": {
        "arms_pools": [
            (_REGIONAL_SECURITY, 0.42), (_CONVENTIONAL_ARMS, 0.25),
            (_SOVEREIGNTY, 0.20), (_VERIFICATION, 0.08), (_NUCLEAR_DISARMAMENT, 0.05),
        ],
        "anchors": ["att"], "anchor_prob": 0.12,
        "arms_paragraphs": (2, 4),
    },
    "ARE": {
        "arms_pools": [
            (_REGIONAL_SECURITY, 0.38), (_CONVENTIONAL_ARMS, 0.28),
            (_SOVEREIGNTY, 0.20), (_VERIFICATION, 0.09), (_NUCLEAR_DISARMAMENT, 0.05),
        ],
        "anchors": ["att"], "anchor_prob": 0.12,
        "arms_paragraphs": (2, 4),
    },
    "IRN": {
        "arms_pools": [
            (_SOVEREIGNTY, 0.38), (_NUCLEAR_DISARMAMENT, 0.30),
            (_REGIONAL_SECURITY, 0.20), (_VERIFICATION, 0.07), (_CONVENTIONAL_ARMS, 0.05),
        ],
        "anchors": ["npt"], "anchor_prob": 0.15,
        "arms_paragraphs": (3, 5),
    },
    "PRK": {
        "arms_pools": [
            (_SOVEREIGNTY, 0.50), (_DETERRENCE, 0.30),
            (_REGIONAL_SECURITY, 0.15), (_NUCLEAR_DISARMAMENT, 0.05),
        ],
        "anchors": [], "anchor_prob": 0.05,
        "arms_paragraphs": (2, 4),
    },
    "ISR": {
        "arms_pools": [
            (_DETERRENCE, 0.35), (_REGIONAL_SECURITY, 0.30),
            (_VERIFICATION, 0.20), (_NUCLEAR_DISARMAMENT, 0.10), (_CONVENTIONAL_ARMS, 0.05),
        ],
        "anchors": ["npt"], "anchor_prob": 0.12,
        "arms_paragraphs": (2, 4),
    },
    "UKR": {
        "arms_pools": [
            (_REGIONAL_SECURITY, 0.35), (_SOVEREIGNTY, 0.25),
            (_NUCLEAR_DISARMAMENT, 0.20), (_CONVENTIONAL_ARMS, 0.15), (_VERIFICATION, 0.05),
        ],
        "anchors": ["npt", "att"], "anchor_prob": 0.22,
        "arms_paragraphs": (3, 5),
    },
    "CUB": {
        "arms_pools": [
            (_SOVEREIGNTY, 0.35), (_NUCLEAR_DISARMAMENT, 0.30),
            (_HUMANITARIAN, 0.20), (_CONVENTIONAL_ARMS, 0.10), (_REGIONAL_SECURITY, 0.05),
        ],
        "anchors": ["tpnw", "npt"], "anchor_prob": 0.30,
        "arms_paragraphs": (3, 5),
    },
}

_DEFAULT_PROFILE = {
    "arms_pools": [
        (_NUCLEAR_DISARMAMENT, 0.35), (_CONVENTIONAL_ARMS, 0.25),
        (_HUMANITARIAN, 0.20), (_REGIONAL_SECURITY, 0.10), (_VERIFICATION, 0.10),
    ],
    "anchors": ["npt", "att"],
    "anchor_prob": 0.22,
    "arms_paragraphs": (2, 4),
}

# ---------------------------------------------------------------------------
# Speech building
# ---------------------------------------------------------------------------

_OPENERS = [
    "Mr. President, I have the honour to address this Assembly on behalf of the Government of {country}. ",
    "Allow me to congratulate you on your election as President of the General Assembly. "
    "The delegation of {country} fully associates itself with the statement made by the Group of {bloc}. ",
    "The delegation of {country} addresses this Assembly at a pivotal moment for international peace and security. ",
    "Madam President, it is my honour to speak on behalf of the people and Government of {country}. ",
    "We gather at a time of profound challenges and opportunities for the international community. "
    "My delegation aligns itself with the statement delivered by the {bloc}. ",
    "The current state of international affairs demands candid and courageous multilateral dialogue. "
    "My delegation wishes to make the following observations on behalf of {country}. ",
]

_CLOSERS = [
    "We call on all Member States to redouble their efforts and demonstrate the political will necessary for progress. "
    "The delegation of {country} stands ready to engage constructively in all relevant processes.",
    "We reaffirm our unwavering commitment to multilateralism as the only viable path to sustainable security. "
    "Only through genuine cooperation can humanity address these profound challenges.",
    "We urge all parties to prioritize dialogue over confrontation in the pursuit of disarmament and development. "
    "{country} will continue to work with all partners toward a more peaceful and equitable world order.",
    "The delegation of {country} thanks the President for the opportunity to address this distinguished body. "
    "We remain committed to constructive engagement and look forward to continued dialogue.",
]

_BLOCS = [
    "Group of 77 and China", "Non-Aligned Movement",
    "G10 Humanitarian Group", "African Group",
    "Latin American and Caribbean Group", "European Union",
    "Nordic countries", "Arab Group",
]


def _pick_weighted(pools: list, rng: np.random.Generator) -> str:
    pool_list, weights = zip(*pools)
    weights = np.array(weights, dtype=float)
    weights /= weights.sum()
    idx = rng.choice(len(pool_list), p=weights)
    return str(rng.choice(pool_list[idx]))


def _build_paragraph(sentences: list) -> str:
    """Join a list of sentence strings into a paragraph."""
    return "  ".join(s.rstrip(". ") + "." for s in sentences)


def _build_full_speech(
    iso3: str,
    country_name: str,
    year: int,
    profile: dict,
    rng: np.random.Generator,
) -> str:
    """
    Build a full General Debate speech of 600-1,500 words.

    Structure:
      - Opening paragraph (procedural)
      - 2-4 non-arms paragraphs (development, HR, general UN)
      - 3-7 arms-control paragraphs (the signal we want to extract)
      - 1-2 non-arms paragraphs
      - Closing paragraph
    """
    paragraphs = []
    filler_pools = [_DEVELOPMENT_FILLER, _HUMAN_RIGHTS_FILLER, _GENERAL_UN_FILLER]

    # --- Opening ---
    opener = str(rng.choice(_OPENERS)).format(
        country=country_name,
        bloc=str(rng.choice(_BLOCS)),
    )
    paragraphs.append(opener)

    # --- Non-arms filler (before arms section) ---
    n_filler_pre = int(rng.integers(2, 5))
    for _ in range(n_filler_pre):
        pool = filler_pools[int(rng.integers(0, len(filler_pools)))]
        sents = [str(rng.choice(pool)) for _ in range(int(rng.integers(2, 5)))]
        paragraphs.append(_build_paragraph(sents))

    # --- Arms control section ---
    n_arms_min, n_arms_max = profile.get("arms_paragraphs", (2, 4))
    n_arms = int(rng.integers(n_arms_min, n_arms_max + 1))
    for i in range(n_arms):
        sents = []
        n_sents = int(rng.integers(3, 7))
        for _ in range(n_sents):
            sents.append(_pick_weighted(profile["arms_pools"], rng))
        # Optionally inject a treaty anchor passage
        if profile["anchors"] and rng.random() < profile["anchor_prob"]:
            treaty = str(rng.choice(profile["anchors"]))
            anchor = _TREATY_ANCHORS.get(treaty, "")
            if anchor:
                sents.append(f"We recall that {anchor[:200].lower().rstrip('.')}.")
        # Add year-specific context in later years
        if year >= 2014 and rng.random() < 0.20:
            sents.append(
                f"The events of the past year — including developments in {year - 1} — "
                f"underscore the urgent need for renewed multilateral commitment."
            )
        paragraphs.append(_build_paragraph(sents))

    # --- Non-arms filler (after arms section) ---
    n_filler_post = int(rng.integers(1, 3))
    for _ in range(n_filler_post):
        pool = filler_pools[int(rng.integers(0, len(filler_pools)))]
        sents = [str(rng.choice(pool)) for _ in range(int(rng.integers(2, 4)))]
        paragraphs.append(_build_paragraph(sents))

    # --- Closing ---
    closer = str(rng.choice(_CLOSERS)).format(country=country_name)
    paragraphs.append(closer)

    return "\n\n".join(paragraphs)


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate_synthetic_corpus(
    countries: Optional[list] = None,
    year_start: int = 2000,
    year_end: int = 2023,
    random_seed: int = 42,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Generate synthetic UNGDC full speeches for arms control analysis.

    Each country-year gets one full speech (600-1,500 words) containing
    arms-control paragraphs embedded within general debate content.
    The segmentation step extracts the arms-relevant portions.

    Returns
    -------
    DataFrame: country_code, country_name, year, session, text, source
    """
    rng = np.random.default_rng(random_seed)
    if countries is None:
        countries = COUNTRY_LIST

    rows = []
    for iso3, name in countries:
        profile = _COUNTRY_PROFILES.get(iso3, _DEFAULT_PROFILE)
        for year in range(year_start, year_end + 1):
            speech = _build_full_speech(iso3, name, year, profile, rng)
            rows.append({
                "country_code": iso3,
                "country_name": name,
                "year": year,
                "session": year - 1945,  # UNGA session number
                "text": speech,
                "source": "synthetic_ungdc",
            })

    df = pd.DataFrame(rows)
    df = df.sort_values(["country_code", "year"]).reset_index(drop=True)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)
        n_words = df["text"].str.split().str.len().sum()
        print(f"  Saved synthetic corpus to {output_path}")
        print(f"  {len(df):,} speeches | {df['country_code'].nunique()} countries "
              f"| {year_start}-{year_end} | ~{n_words:,} total words")

    return df


if __name__ == "__main__":
    out = Path(__file__).parent.parent.parent / "data" / "synthetic" / "ungdc_synthetic.parquet"
    df = generate_synthetic_corpus(output_path=out)
    print(df[["country_code", "year"]].describe())
    print("Sample speech (first 400 chars):")
    print(df["text"].iloc[0][:400])
