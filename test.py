# Final Capstone Project — CT RISE Grant Matcher (v1 OpenAI tool-enabled)

import os, json, re, time, pandas as pd, streamlit as st, openai
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

# ─────────────────────── CONFIG ───────────────────────
SEARCH_MODEL = "gpt-4o-mini-search-preview"    # or gpt-4o-search-preview
CHAT_MODEL   = "gpt-3.5-turbo"
EMB_MODEL    = "text-embedding-ada-002"
NEEDED       = 10
MAX_RETRY    = 4
BACKOFF      = 2

MISSION = (
    "The Connecticut RISE Network empowers public high schools with data-driven strategies "
    "and personalized support to improve student outcomes and promote post-secondary success, "
    "especially for Black, Latinx, and low-income youth."
)

# ─────────────────────── KEYS ─────────────────────────
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# ─────────────────────── HELPERS ──────────────────────
def retry(fn):
    def wrapper(*args, **kw):
        for i in range(MAX_RETRY):
            try:
                return fn(*args, **kw)
            except openai.RateLimitError:
                time.sleep(BACKOFF * (i + 1))
        st.error("OpenAI rate-limit; try again later."); st.stop()
    return wrapper

@retry
def embed(text):
    return openai.embeddings.create(model=EMB_MODEL, input=text).data[0].embedding

@retry
def chat(model, messages, **kw):
    return openai.chat.completions.create(model=model, messages=messages, **kw)

# ─── STEP 1 — single search call ──────────────────────
def fetch_grants():
    system = {"role": "system", "content": "You are a meticulous grants researcher."}
    user   = {
        "role": "user",
        "content": (
            f"Search the web and return exactly {NEEDED} CURRENT US grant opportunities for "
            "nonprofits working on high-school education, youth equity, or college readiness. "
            "Return STRICT JSON array; keys: title, sponsor, amount, deadline, url, summary."
        ),
        "tool_choice": "auto"
    }

    response = chat(
        model=SEARCH_MODEL,
        messages=[system, user],
        tools=[{"type": "search"}],          # enable internal search tool
        response_format={"type": "json_object"},
        max_tokens=1400,
        temperature=0.3,
    )

    raw_json = response.choices[0].message.content
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError:
        m = re.search(r"\[.*\]", raw_json, re.S)
        data = json.loads(m.group()) if m else []
    cleaned = [
        {k: d.get(k, "N/A")
         for k in ("title", "sponsor", "amount", "deadline", "url", "summary")}
        for d in data
    ]
    return cleaned[:NEEDED]

# ─── STEP 2 — rank & add “Why Fit” ────────────────────
def build_table(rows):
    df = pd.DataFrame(rows)
    mission_vec = embed(MISSION)
    df["%Match"] = (
        df.summary.apply(lambda t: cosine_similarity([embed(t)], [mission_vec])[0][0] * 100)
        .round(1)
    )
    df = df.sort_values("%Match", ascending=False).reset_index(drop=True)
    df.index += 1
    df.insert(0, "Rank", df.index)

    whys = []
    for _, r in df.iterrows():
        q = (f'In one sentence: why does the grant titled "{r.title}" align with the mission '
             f'"{MISSION}"?')
        a = chat(model=CHAT_MODEL,
                 messages=[{"role": "user", "content": q}],
                 max_tokens=60, temperature=0.3).choices[0].message.content.strip()
        whys.append(a)
    df["Why Fit"] = whys
    return df

# ─── STREAMLIT UI ────────────────────────────────────
st.title("Final Capstone Project (Grant Matcher for CT RISE – GPT-4o Search)")
st.write("> **Mission:**", MISSION)

if st.button("🚀 Generate & Rank 10 Grants", type="primary"):
    with st.spinner("GPT-4o searching web and compiling grants…"):
        grants = fetch_grants()
        if len(grants) < NEEDED:
            st.error("Model returned fewer than 10 grants. Click again.")
        else:
            st.session_state["tbl"] = build_table(grants)
            st.success("Done!")

if "tbl" in st.session_state:
    st.dataframe(
        st.session_state["tbl"][[
            "Rank", "title", "sponsor", "amount", "deadline",
            "%Match", "Why Fit", "url", "summary"
        ]],
        use_container_width=True,
    )
else:
    st.caption("Press the rocket to generate a ranked list.")
