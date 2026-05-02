"""
Additional styles and variation pools for Phase 2:
- Toddler Autism Social Media Posts
- Diabetes IoT Sensor Logs
"""
import random
import json
from datetime import datetime, timedelta

# =======================================================================
# TODDLER AUTISM — SOCIAL MEDIA POSTS
# =======================================================================

FORUMS = [
    "r/AutismParenting", "r/Toddlers", "Facebook: Autism Moms Support Group",
    "Mumsnet: Special Needs", "Reddit: r/DevelopmentalDelays",
    "BabyCenter: Worried about Autism"
]

USERS = [
    "worried_mom_99", "anxious_papa", "sarah_bear", "mama_bear_1990",
    "confused_dad", "toddler_mom_xoxo", "user109283", "anon_parent",
    "jessica_m", "david_k"
]

HASHTAGS = [
    "#autism #toddler #help", "#ASD #momlife", "#developmental #speechdelay",
    "#parenting #autismawareness", "#worried", "#advice #toddlers"
]

# Q-Chat 10 representations in Social Media Slang
# For each item, we need a phrase for "Yes (traits)" and "No (typical)"
# Keep in mind 1 means traits present, 0 means typical.
QCHAT_SOCIAL = {
    "A1": { # Looks when name called
        1: ["he completely ignores me when I call his name", "never looks at me when I call her name", "doesn't respond to his name AT ALL", "I call and call but no response"],
        0: ["responds to his name right away", "always looks at me when I call her", "knows his name perfectly"]
    },
    "A2": { # Eye contact
        1: ["zero eye contact", "never looks me in the eyes", "avoids eye contact", "won't meet my gaze"],
        0: ["makes great eye contact", "looks me in the eyes", "eye contact is totally fine"]
    },
    "A3": { # Pointing for wants
        1: ["doesn't point when he wants something", "just grabs my hand to get things instead of pointing", "never points at what she wants"],
        0: ["points at his cup when he wants it", "points to things she wants", "always pointing to ask for things"]
    },
    "A4": { # Pointing for interest
        1: ["never points to show me a bird or airplane", "doesn't point to share things with me", "no pointing to show interest"],
        0: ["points out dogs and cars to me all the time", "loves pointing at airplanes to show me", "always points to share cool stuff"]
    },
    "A5": { # Pretend play
        1: ["doesn't do any pretend play", "just lines up his toys, no pretending", "won't feed his teddy bear or play pretend"],
        0: ["loves playing pretend with her kitchen", "feeds his stuffed animals", "great imagination, lots of pretend play"]
    },
    "A6": { # Follows gaze
        1: ["if I look at something across the room, he doesn't follow my gaze", "doesn't look where I look", "never tracks what I'm looking at"],
        0: ["always looks where I point or look", "follows my gaze across the room", "tracks what I'm looking at"]
    },
    "A7": { # Comforts others
        1: ["doesn't seem to care if I'm crying", "shows no empathy when someone is hurt", "doesn't try to comfort his sister when she cries"],
        0: ["always brings me a toy if I'm upset", "very empathetic, comforts his brother", "gets sad when I cry and hugs me"]
    },
    "A8": { # First words typical
        1: ["still no words yet", "speech is super delayed", "only babbles, no real words", "lost the few words he had"],
        0: ["talking up a storm", "vocabulary is exploding", "says so many words now!"]
    },
    "A9": { # Simple gestures (wave)
        1: ["doesn't wave bye-bye or clap", "no waving at all", "still not clapping or waving"],
        0: ["waves bye-bye to everyone", "loves clapping his hands", "waves and claps all the time"]
    },
    "A10": { # Stares at nothing
        1: ["stares blankly at the ceiling fan for hours", "zones out and stares at nothing", "gets mesmerized by spinning wheels"],
        0: ["doesn't really zone out", "doesn't stare blankly", "no unusual staring behaviors"]
    }
}

def toddler_social_media_post(row, rid):
    """Generate a realistic social media post."""
    age = row["Age_Mons"]
    sex = str(row["Sex"]).strip().lower()
    pronoun = "he" if sex == 'm' else "she"
    child = "son" if sex == 'm' else "daughter"
    
    forum = random.choice(FORUMS)
    user = random.choice(USERS)
    hashtag = random.choice(HASHTAGS)
    
    # Select 3-5 random traits to actually mention in the text
    # (Real parents don't list all 10 in a paragraph, they list a few)
    # But wait! For the parser to work, ALL 10 need to be present OR
    # we need to ensure the unmentioned ones default to 0. 
    # To make extraction 100% accurate as requested, we MUST mention all 10 items.
    
    # We will weave all 10 items naturally into a rant.
    items = [f"A{i}" for i in range(1, 11)]
    random.shuffle(items)
    
    sentences = []
    for k in items:
        val = int(row[k])
        sentences.append(random.choice(QCHAT_SOCIAL[k][val]))
        
    # Combine into a rambling paragraph
    paragraph = " ".join(sentences).capitalize()
    
    # Typos in social media
    if random.random() < 0.2:
        paragraph = paragraph.replace("because", "bc").replace("with", "w/").replace("you", "u")
        
    # Jaundice & Family ASD
    jaundice = "Yes" if str(row.get("Jaundice", "no")).strip().lower() == "yes" else "No"
    fam = "Yes" if str(row.get("Family_mem_with_ASD", "no")).strip().lower() == "yes" else "No"
    
    j_text = "He had jaundice as a baby." if jaundice == "Yes" else "No newborn jaundice."
    f_text = "Autism runs in our family." if fam == "Yes" else "No family history of ASD at all."
    
    ethnicity = str(row.get("Ethnicity", "Unknown")).strip()
    completed_by = str(row.get("Who completed the test", "family member")).strip()
    
    post = f"""=== POST ID: {rid:05d} ===
Platform: {forum}
User: {user}
Posted: {random.randint(1,24)} hours ago

Hi everyone, I'm really worried about my {age} month old {child}. {j_text} {f_text}
{paragraph}.

Does this sound like autism? I ({completed_by}, {ethnicity}) am freaking out. Any advice? {hashtag}
"""
    return post


# =======================================================================
# DIABETES — IOT SENSOR LOGS
# =======================================================================

DEVICES = ["CGM-X200", "WEARABLE-METABOLIC-v4", "GLUCO-TRACKER-PRO", "SMART-PUMP-900"]

def diabetes_style_iot_log(row, rid, symptom_cols):
    """Generate a JSON/Syslog style IoT output."""
    age = row["Age"]
    gender = str(row["Gender"]).strip()
    g_abbr = "M" if gender == "Male" else "F"
    is_pos = str(row["class"]).strip() == "Positive"
    
    device = random.choice(DEVICES)
    timestamp = (datetime.now() - timedelta(days=random.randint(1, 365))).strftime("%Y-%m-%dT%H:%M:%SZ")
    
    present = [c.upper().replace(" ", "_") for c in symptom_cols if str(row[c]).strip().lower() == "yes"]
    
    log_format = random.choice(["JSON", "SYSLOG"])
    
    if log_format == "JSON":
        data = {
            "log_id": f"SEQ-{rid:05d}",
            "timestamp": timestamp,
            "device_id": device,
            "patient": {
                "age": int(age),
                "gender": g_abbr
            },
            "telemetry": {
                "symptoms_flagged": present,
                "risk_classification": "HIGH" if is_pos else "LOW"
            }
        }
        return json.dumps(data, indent=2)
    else:
        # Syslog style
        sym_str = ",".join(present) if present else "NONE"
        risk = "HIGH" if is_pos else "LOW"
        return f"[{timestamp}] SYS_LOG | DEV={device} | PT_AGE={age} | GENDER={g_abbr} | SYMPT_FLAG={sym_str} | RISK_CLASS={risk}"
