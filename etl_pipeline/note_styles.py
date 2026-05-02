"""
Variation pools and style engines for realistic clinical note generation.
Used by generate_unstructured.py to produce genuinely unstructured text.
"""
import random

# ── Doctor names & clinic noise ─────────────────────────────────────────
DOCTORS = [
    "Dr. R. Mansour", "Dr. S. El-Amin", "Dr. K. Haddad", "Dr. A. Farouk",
    "Dr. N. Ibrahim", "Dr. L. Osman", "Dr. T. Khalil", "Dr. M. Barakat",
    "Dr. H. Salem", "Dr. J. Nasser", "Dr. P. Reeves", "Dr. W. Chen",
]

NOISE_AUTISM = [
    "Pt was cooperative throughout the session.",
    "Assessment conducted in presence of the mother.",
    "Child appeared well-nourished, no acute distress.",
    "Good eye contact maintained during interview.",
    "Parent reports child is performing well academically.",
    "Sleep patterns reported as normal by caregiver.",
    "No concerns regarding gross motor development.",
    "Speech and language development WNL for age.",
    "Currently enrolled in mainstream schooling.",
    "No medications currently prescribed.",
    "Vital signs WNL.", "Pt accompanied by both parents.",
    "Behavioral observation: calm, engaged, follows instructions.",
    "No history of seizures or head trauma.",
    "Immunizations up to date per caregiver report.",
]

NOISE_DIABETES = [
    "Pt appears well-nourished.", "No acute distress noted.",
    "Vitals: BP 130/85, HR 78, Temp 37.0C.",
    "Vitals: BP 142/90, HR 82, Temp 36.8C, RR 16.",
    "BMI not calculated at this visit.",
    "Patient counseled on dietary modifications.",
    "Follow-up labs ordered: FBG, HbA1c, lipid panel.",
    "Pt reports family hx of T2DM in first-degree relative.",
    "Social hx: non-smoker, occasional alcohol use.",
    "Medication reconciliation performed.",
    "Pt educated on warning signs of hyperglycemia.",
    "Reviewed importance of regular physical activity.",
    "No known drug allergies (NKDA).",
]

NOISE_STROKE = [
    "Vitals on admission: BP 158/92, HR 88, O2 sat 96% RA.",
    "Neurological exam: cranial nerves II-XII grossly intact.",
    "NIHSS score was not formally calculated at this visit.",
    "Patient alert and oriented x3.",
    "Medication list reviewed and reconciled.",
    "Discussed stroke prevention strategies with patient.",
    "Follow-up with neurology in 3 months.",
    "PT/OT referral placed.", "Social work consult requested.",
    "Fall risk assessment: moderate.",
    "Diet: cardiac / low-sodium as tolerated.",
    "DVT prophylaxis initiated.",
]

# ── TYPO INJECTION ──────────────────────────────────────────────────────
# Only applied to ~5% of records; typos on non-critical descriptive words
TYPO_MAP = {
    "screening": ["screeening", "screenng"],
    "behavioral": ["behavorial", "behaviorial"],
    "assessment": ["assesment", "assessement"],
    "developmental": ["developemental", "developmetal"],
    "spectrum": ["specturm", "spectum"],
    "threshold": ["threshhold", "treshold"],
    "evaluation": ["evalution", "evaluaiton"],
    "questionnaire": ["questionaire", "questionairre"],
    "consultation": ["consultaiton", "consulation"],
    "cerebrovascular": ["cerebrovasular", "cerebrovasclar"],
    "hypertension": ["hypertesion", "hypertenion"],
    "endocrinology": ["endocrinolgy", "endocrinoloy"],
    "polydipsia": ["polydpsia", "polydipisa"],
    "polyuria": ["polyurea", "polyura"],
    "polyphagia": ["polyphaiga", "polyphgia"],
    "neonatal": ["neonatal", "neonatal"],  # keep safe
}

def inject_typos(text: str) -> str:
    """Randomly inject 1-2 typos into text."""
    words = list(TYPO_MAP.keys())
    random.shuffle(words)
    count = 0
    for w in words:
        if w in text.lower() and count < 2:
            typo = random.choice(TYPO_MAP[w])
            text = text.replace(w, typo, 1)
            count += 1
    return text

# ── AGE+SEX PHRASING ────────────────────────────────────────────────────
def age_sex_phrase(age, sex_str: str) -> str:
    """Return varied age+sex description."""
    s = sex_str.lower()
    full = "male" if s in ("m", "male") else "female"
    abbr = "M" if full == "male" else "F"
    child = "boy" if full == "male" else "girl"
    phrases = [
        f"{age}-year-old {full}",
        f"{age} yo {abbr}",
        f"{full}, {age} years old",
        f"{age}-yr-old {full}",
        f"{age}y/o {abbr}",
        f"{full} patient, age {age}",
        f"{age} year old {child}",
        f"{child}, aged {age}",
    ]
    return random.choice(phrases)

# ── JAUNDICE PHRASING ───────────────────────────────────────────────────
def jaundice_phrase(has_jaundice: bool) -> str:
    if has_jaundice:
        return random.choice([
            "History of neonatal jaundice.",
            "Neonatal jaundice: positive.",
            "Hx neonatal jaundice noted.",
            "Born with jaundice per parent report.",
            "Neonatal icterus documented.",
            "jaundice(+) in neonatal period.",
            "Neonatal jaundice present.",
            "Jaundice hx: yes.",
        ])
    return random.choice([
        "No history of neonatal jaundice.",
        "Neonatal jaundice: negative.",
        "Denies neonatal jaundice.",
        "No neonatal icterus.",
        "jaundice(-)",
        "No jaundice reported in neonatal period.",
        "Jaundice hx: no.",
        "Neonatal period unremarkable for jaundice.",
    ])

# ── FAMILY ASD PHRASING ─────────────────────────────────────────────────
def family_asd_phrase(has_fam: bool) -> str:
    if has_fam:
        return random.choice([
            "Positive family history of ASD.",
            "Family hx of autism spectrum disorder reported.",
            "Fam hx ASD: positive.",
            "Family member(s) diagnosed with ASD.",
            "Significant family hx includes ASD.",
            "First-degree relative with autism dx.",
            "Family autism history: yes.",
            "ASD runs in the family per parent.",
        ])
    return random.choice([
        "No family history of ASD.",
        "Family hx of ASD: negative.",
        "Fam hx ASD: unremarkable.",
        "No known family members with ASD.",
        "Denies family history of autism.",
        "Family hx: no ASD.",
        "Family autism history: no.",
        "No ASD in family per parent report.",
    ])

# ── AQ-10 PRESENTATION STYLES ───────────────────────────────────────────
AQ_DESCRIPTIONS = {
    "A1": "notices small sounds others miss",
    "A2": "concentrates on whole picture vs details",
    "A3": "multitasking ability",
    "A4": "switching back after interruption",
    "A5": "reading between the lines",
    "A6": "detecting if listener is bored",
    "A7": "difficulty with characters intentions in stories",
    "A8": "collecting info about categories",
    "A9": "reading emotions from faces",
    "A10": "difficulty understanding peoples intentions",
}

AQ_ITEMS_FULL = {
    "A1": "I often notice small sounds when others do not",
    "A2": "I usually concentrate more on the whole picture rather than the small details",
    "A3": "I find it easy to do more than one thing at once",
    "A4": "If there is an interruption I can switch back to what I was doing very quickly",
    "A5": "I find it easy to read between the lines when someone is talking to me",
    "A6": "I know how to tell if someone listening to me is getting bored",
    "A7": "When I'm reading a story I find it difficult to work out the characters' intentions",
    "A8": "I like to collect information about categories of things",
    "A9": "I find it easy to work out what someone is thinking or feeling just by looking at their face",
    "A10": "I find it difficult to work out people's intentions",
}

def aq10_style_peritem(vals: dict) -> str:
    """Style 1: Per-item list with varied wording."""
    lines = ["AQ-10 Screening Results:"]
    for i in range(1, 11):
        k = f"A{i}"
        v = vals[k]
        style = random.randint(0, 2)
        if style == 0:
            resp = "endorsed (YES)" if v == 1 else "not endorsed (NO)"
            lines.append(f"    Item {k}: \"{AQ_ITEMS_FULL[k]}\" — {resp}")
        elif style == 1:
            resp = "YES" if v == 1 else "NO"
            lines.append(f"    {k}: {resp} — {AQ_DESCRIPTIONS[k]}")
        else:
            resp = "(+)" if v == 1 else "(-)"
            lines.append(f"    {k} {resp}")
    score = sum(vals[f"A{i}"] for i in range(1, 11))
    lines.append(f"  Total AQ-10 Score: {score} out of 10")
    return "\n".join(lines)

def aq10_style_compact(vals: dict) -> str:
    """Style 2: Compact one-line format."""
    parts = []
    for i in range(1, 11):
        k = f"A{i}"
        sym = "+" if vals[k] == 1 else "-"
        parts.append(f"{k}({sym})")
    score = sum(vals[f"A{i}"] for i in range(1, 11))
    return f"AQ-10: {' '.join(parts)} = {score}/10"

def aq10_style_endorsed_list(vals: dict) -> str:
    """Style 3: Lists endorsed and not endorsed items."""
    endorsed = [f"A{i}" for i in range(1, 11) if vals[f"A{i}"] == 1]
    not_endorsed = [f"A{i}" for i in range(1, 11) if vals[f"A{i}"] == 0]
    score = len(endorsed)
    lines = [f"AQ-10 administered. Score: {score}/10."]
    if endorsed:
        lines.append(f"  Endorsed items: {', '.join(endorsed)}.")
    else:
        lines.append("  No items endorsed.")
    if not_endorsed:
        lines.append(f"  Not endorsed: {', '.join(not_endorsed)}.")
    return "\n".join(lines)

def aq10_style_narrative(vals: dict) -> str:
    """Style 4: Narrative description of endorsed items."""
    endorsed = [f"A{i}" for i in range(1, 11) if vals[f"A{i}"] == 1]
    score = len(endorsed)
    if not endorsed:
        desc = "The patient did not endorse any of the AQ-10 items."
    elif len(endorsed) <= 3:
        descs = [f"{k} ({AQ_DESCRIPTIONS[k]})" for k in endorsed]
        desc = f"Patient endorsed {', '.join(descs)}. Remaining items not endorsed."
    else:
        descs = [f"{k} ({AQ_DESCRIPTIONS[k]})" for k in endorsed]
        desc = f"Patient endorsed the following: {'; '.join(descs)}. All other items denied."
    return f"AQ-10 was administered.\n  {desc}\n  Total score: {score} out of 10."

def aq10_style_table(vals: dict) -> str:
    """Style 5: Key=value format."""
    lines = ["AQ-10 Item Responses:"]
    for i in range(1, 11):
        k = f"A{i}"
        lines.append(f"  {k} = {vals[k]}")
    score = sum(vals[f"A{i}"] for i in range(1, 11))
    lines.append(f"  AQ-10 Total = {score}")
    return "\n".join(lines)

def aq10_block(vals: dict) -> str:
    """Randomly pick an AQ-10 presentation style."""
    return random.choice([
        aq10_style_peritem,
        aq10_style_compact,
        aq10_style_endorsed_list,
        aq10_style_narrative,
        aq10_style_table,
    ])(vals)

# ── CLASSIFICATION PHRASING ──────────────────────────────────────────────
def asd_class_phrase(is_positive: bool) -> str:
    if is_positive:
        return random.choice([
            "Classification: YES\n  ASD POSITIVE — referral for comprehensive evaluation recommended.",
            "Screen result: POSITIVE for ASD. Further assessment indicated.",
            "Clinical impression: ASD screen positive. Refer to specialist.",
            "ASD screening: POSITIVE (score >= 6). Recommend full diagnostic workup.",
            "Assessment: ASD likely, referral for formal dx initiated.",
            "Result: ASD positive. Comprehensive evaluation ordered.",
            "Impression: Positive ASD screen. Referral recommended.",
        ])
    return random.choice([
        "Classification: NO\n  ASD NEGATIVE — no further evaluation indicated at this time.",
        "Screen result: NEGATIVE for ASD.",
        "Clinical impression: ASD screen negative. No referral needed.",
        "ASD screening: NEGATIVE (score < 6). Routine follow-up.",
        "Assessment: ASD unlikely at this time.",
        "Result: ASD negative. No further action.",
        "Impression: Negative ASD screen. Continue routine monitoring.",
    ])

def diabetes_class_phrase(is_positive: bool) -> str:
    if is_positive:
        return random.choice([
            "Risk Classification: Positive\n  POSITIVE — meets criteria for early-stage diabetes risk. FBG and HbA1c recommended.",
            "Diabetes Risk: POSITIVE. Further workup strongly advised.",
            "Impression: High risk for diabetes. Order FBG, HbA1c, OGTT.",
            "Dx: Positive diabetes risk screening. Refer to endocrinology.",
            "Assessment: Diabetes risk - POSITIVE. Labs ordered.",
            "Clinical decision: Positive screen. Initiate diabetic workup.",
        ])
    return random.choice([
        "Risk Classification: Negative\n  NEGATIVE — does not meet criteria for diabetes risk. Routine follow-up advised.",
        "Diabetes Risk: NEGATIVE. No immediate workup needed.",
        "Impression: Low risk for diabetes at this time.",
        "Dx: Negative diabetes risk screening. Routine f/u.",
        "Assessment: Diabetes risk - NEGATIVE.",
        "Clinical decision: Negative screen. Continue annual screening.",
    ])

def stroke_class_phrase(had_stroke: bool) -> str:
    if had_stroke:
        return random.choice([
            "Outcome: Stroke\n  STROKE OCCURRED — cerebrovascular event confirmed. Neuro intervention initiated.",
            "CVA Status: STROKE confirmed. Thrombolysis administered.",
            "Cerebrovascular event: YES. Acute ischemic stroke documented.",
            "Dx: Stroke. Patient admitted to stroke unit.",
            "Event: Stroke occurred during observation period.",
            "Impression: Acute stroke. Code stroke activated.",
        ])
    return random.choice([
        "Outcome: No Stroke\n  NO STROKE — no cerebrovascular event during observation.",
        "CVA Status: No stroke. No acute neurological deficit.",
        "Cerebrovascular event: NO. No stroke during observation period.",
        "Dx: No stroke. Discharged with preventive counseling.",
        "Event: No stroke occurred.",
        "Impression: No cerebrovascular event. Continue risk factor management.",
    ])

# ── DIABETES SYMPTOM PHRASING ────────────────────────────────────────────
SYMPTOM_SYNONYMS = {
    "Polyuria": ["excessive urination", "polyuria", "frequent urination", "increased urinary freq"],
    "Polydipsia": ["excessive thirst", "polydipsia", "increased thirst", "constant thirst"],
    "sudden weight loss": ["sudden weight loss", "unexplained wt loss", "rapid weight decrease", "unintentional weight loss"],
    "weakness": ["weakness", "generalized fatigue", "feeling weak and tired", "fatigue and weakness"],
    "Polyphagia": ["excessive hunger", "polyphagia", "increased appetite", "constant hunger"],
    "Genital thrush": ["genital thrush", "recurrent genital yeast infections", "genital candidiasis", "thrush infections"],
    "visual blurring": ["visual blurring", "blurred vision", "blurry vision episodes", "visual disturbance"],
    "Itching": ["itching", "persistent pruritus", "skin itching", "generalized itching"],
    "Irritability": ["irritability", "mood changes", "irritable mood", "increased irritability"],
    "delayed healing": ["delayed healing", "slow wound healing", "wounds heal slowly", "poor wound healing"],
    "partial paresis": ["partial paresis", "muscle weakness in extremities", "limb weakness", "partial paralysis"],
    "muscle stiffness": ["muscle stiffness", "joint stiffness", "stiff muscles and joints", "musculoskeletal stiffness"],
    "Alopecia": ["alopecia", "hair loss", "hair thinning", "losing hair"],
    "Obesity": ["obesity", "clinically obese", "BMI > 30", "obese"],
}

def diabetes_symptoms_block(row, symptom_cols: list) -> str:
    """Generate varied symptom presentation."""
    present = [c for c in symptom_cols if str(row[c]).strip().lower() == "yes"]
    absent = [c for c in symptom_cols if str(row[c]).strip().lower() == "no"]
    style = random.randint(0, 3)

    if style == 0:  # Bullet list
        lines = []
        if present:
            lines.append("  Reports the following symptoms:")
            for s in present:
                lines.append(f"    - {random.choice(SYMPTOM_SYNONYMS[s])}")
        if absent:
            lines.append("  Denies:")
            for s in absent:
                lines.append(f"    - {random.choice(SYMPTOM_SYNONYMS[s])}")
        return "\n".join(lines)

    elif style == 1:  # Compact +/-
        parts = []
        for s in symptom_cols:
            syn = random.choice(SYMPTOM_SYNONYMS[s])
            sym = "+" if s in present else "-"
            parts.append(f"{syn}({sym})")
        return "  Symptom review: " + ", ".join(parts)

    elif style == 2:  # Narrative
        lines = []
        if present:
            descs = [random.choice(SYMPTOM_SYNONYMS[s]) for s in present]
            if len(descs) <= 3:
                lines.append(f"  Patient complains of {', '.join(descs)}.")
            else:
                lines.append(f"  Patient presents with multiple complaints including {', '.join(descs[:3])},")
                lines.append(f"  as well as {', '.join(descs[3:])}.")
        else:
            lines.append("  Patient denies all screened symptoms.")
        if absent and present:
            lines.append(f"  Specifically denies {', '.join(random.choice(SYMPTOM_SYNONYMS[s]) for s in absent[:3])}.")
            if len(absent) > 3:
                lines.append("  Remainder of review of systems negative.")
        elif absent and not present:
            lines.append("  All screened symptoms denied.")
        return "\n".join(lines)

    else:  # Present-only + blanket denial
        lines = []
        if present:
            lines.append("  Positive findings:")
            for s in present:
                lines.append(f"    • {random.choice(SYMPTOM_SYNONYMS[s])}")
        if absent:
            if len(absent) == len(symptom_cols):
                lines.append("  All screened symptoms denied.")
            else:
                lines.append("  All other screened symptoms denied.")
        return "\n".join(lines)

# ── STROKE FIELD PHRASING ────────────────────────────────────────────────
def stroke_demographics_block(row) -> str:
    gender = str(row["gender"]).strip()
    age = row["age"]
    pronoun = "He" if gender == "Male" else "She"
    married = str(row["ever_married"]).strip()
    work = str(row["work_type"]).strip()
    residence = str(row["Residence_type"]).strip()

    style = random.randint(0, 3)
    if style == 0:
        married_t = f"{pronoun} is currently or has been previously married." if married == "Yes" else f"{pronoun} has never been married."
        return (f"  {gender} patient, {age} years of age.\n"
                f"  {married_t}\n"
                f"  Employment: {work}.\n"
                f"  Residence: {residence.lower()} area.")
    elif style == 1:
        m_abbr = "married" if married == "Yes" else "single/never married"
        g_abbr = "M" if gender == "Male" else "F"
        return (f"  {age}yo {g_abbr}, {m_abbr}\n"
                f"  Work: {work} | Lives in {residence.lower()} area")
    elif style == 2:
        married_t = "married" if married == "Yes" else "never married"
        return (f"  Demographics: {gender}, {age} years old, {married_t}\n"
                f"  Employment status: {work} sector. Resides in {residence.lower()} area.")
    else:
        married_t = f"{pronoun} has been married." if married == "Yes" else f"Not married."
        return (f"  Patient is a {age}-year-old {gender.lower()}.\n"
                f"  {married_t} Works in {work}.\n"
                f"  Lives in an {residence.lower()} area.")

def stroke_comorbid_block(row) -> str:
    htn = int(row["hypertension"])
    hd = int(row["heart_disease"])
    style = random.randint(0, 2)
    if htn == 0 and hd == 0:
        return random.choice([
            "  No significant comorbidities (no hypertension, no heart disease).",
            "  PMH: unremarkable. No HTN, no cardiac history.",
            "  Medical history: no hypertension, no heart disease.",
            "  Comorbidities: none. HTN(-), heart disease(-)",
        ])
    parts_formal = []
    parts_abbr = []
    if htn:
        parts_formal.append("documented hypertension")
        parts_abbr.append("HTN")
    if hd:
        parts_formal.append("history of heart disease")
        parts_abbr.append("heart dz")
    if style == 0:
        return f"  Significant comorbidities include: {', '.join(parts_formal)}."
    elif style == 1:
        return f"  PMH: {', '.join(parts_abbr)}."
    else:
        return f"  Medical history significant for {', '.join(parts_formal)}."

def stroke_measurements_block(row) -> str:
    import pandas as pd
    glucose = row["avg_glucose_level"]
    bmi = row["bmi"]
    smoking = str(row["smoking_status"]).strip()
    style = random.randint(0, 2)
    bmi_str = f"{bmi}" if pd.notna(bmi) else "not recorded"
    if style == 0:
        bmi_line = f"BMI: {bmi} kg/m²" if pd.notna(bmi) else "BMI: Not recorded (missing value)"
        return (f"  Average glucose level: {glucose} mg/dL\n"
                f"  {bmi_line}\n"
                f"  Smoking history: {smoking}.")
    elif style == 1:
        bmi_part = f"BMI {bmi}" if pd.notna(bmi) else "BMI N/A"
        return f"  Labs/Vitals: glucose {glucose} mg/dL, {bmi_part} kg/m², smoking: {smoking}"
    else:
        bmi_line = f"Body mass index: {bmi} kg/m²" if pd.notna(bmi) else "BMI not available"
        return (f"  Glucose (avg): {glucose} mg/dL\n"
                f"  {bmi_line}\n"
                f"  Tobacco use: {smoking}.")

# ── NOTE HEADERS ─────────────────────────────────────────────────────────
HEADERS_AUTISM = [
    "AUTISM SCREENING — CLINICAL ASSESSMENT NOTE",
    "BEHAVIORAL SCREENING REPORT — AQ-10 EVALUATION",
    "DEVELOPMENTAL SCREENING DOCUMENTATION",
    "ASD EVALUATION — BEHAVIORAL HEALTH",
    "NEURODEVELOPMENTAL SCREENING NOTE",
    "PEDIATRIC DEVELOPMENTAL ASSESSMENT",
    "BEHAVIORAL HEALTH — AQ-10 SCREENING",
    "CLINICAL NOTE — AUTISM SPECTRUM SCREENING",
]

HEADERS_DIABETES = [
    "DIABETES RISK ASSESSMENT — CLINICAL INTAKE NOTE",
    "ENDOCRINOLOGY CONSULTATION — SYMPTOM EVALUATION",
    "PRIMARY CARE ASSESSMENT — METABOLIC SCREENING",
    "DIABETES SCREENING — OUTPATIENT NOTE",
    "METABOLIC RISK EVALUATION — CLINICAL NOTE",
    "INTERNAL MEDICINE — DIABETES RISK ASSESSMENT",
]

HEADERS_STROKE = [
    "CEREBROVASCULAR RISK ASSESSMENT — DISCHARGE SUMMARY",
    "NEUROLOGY DEPARTMENT — STROKE RISK EVALUATION",
    "CLINICAL CASE SUMMARY — CARDIOVASCULAR PROFILE",
    "STROKE UNIT — PATIENT EVALUATION",
    "NEUROLOGY CONSULT — CVA RISK ASSESSMENT",
    "CEREBROVASCULAR SCREENING — CLINICAL NOTE",
]
