# Final Capstone Project (Grant Matcher for CT RISE — GPT-4o Search)

import os, json, re, time, pandas as pd, streamlit as st, openai
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

# ---------- CONFIG -------------------------------------------------
MODEL_SEARCH = "gpt-4o-mini-search-preview"   # shown in your quota screen
MODEL_CHAT   = "gpt-3.5-turbo"                # for short follow-ups
EMB_MODEL    = "text-embedding-ada-002"
N_GRANTS     = 10
OPENAI_RETRY = 4
BACKOFF      = 2                              # seconds

MISSION = (
    "The Connecticut RISE Network empowers public high schools with data-driven strategies "
    "and personalized support to improve student outcomes and promote postsecondary success, "
    "especially for Black, Latinx, and low-income youth."
)

# ---------- KEYS ---------------------------------------------------
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# ---------- OPENAI WRAPPERS ---------------------------------------
def chat(model, messages, **kwargs):
    for a in range(OPENAI_RETRY):
        try:
            resp = openai.ChatCompletion.create(model=model,
                                                messages=messages,
                                                temperature=0.25,
                                                **kwargs)
            return resp.choices[0].message.content
        except openai.error.RateLimitError:
            time.sleep(BACKOFF * (a + 1))
    st.error("OpenAI rate-limit hit repeatedly."); st.stop()

def embed(text):
    for a in range(OPENAI_RETRY):
        try:
            return openai.Embedding.create(input=text, model=EMB_MODEL)["data"][0]["embedding"]
        except openai.error.RateLimitError:
            time.sleep(BACKOFF * (a + 1))
    st.error("OpenAI embedding rate-limit."); st.stop()

# ---------- STEP 1: single GPT-4o search call ----------------------
def get_grants():
    user = (
        f"Using your web-search capabilities, provide EXACTLY {N_GRANTS} current grant "
        "opportunities suitable for US nonprofits focused on high-school education, youth equity "
        "or college readiness. Return ONLY a valid JSON array; each object MUST have keys "
        "title, sponsor, amount, deadline, url, summary."
    )
    raw = chat(MODEL_SEARCH,
               [{"role":"user","content": user}],
               response_format={"type":"json_object"},
               max_tokens=1500)

    # robust parse
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", raw, re.S)
        data = json.loads(match.group()) if match else []
    cleaned = []
    for d in data:
        cleaned.append({k: d.get(k, "N/A") for k in
                        ("title","sponsor","amount","deadline","url","summary")})
    return cleaned[:N_GRANTS]

# ---------- STEP 2: rank & add Why-Fit -----------------------------
def build_table(rows):
    df = pd.DataFrame(rows)
    mvec = embed(MISSION)
    df["%Match"] = (
        df.summary.apply(lambda s: cosine_similarity([embed(s)], [mvec])[0][0]*100)
        .round(1)
    )
    df = df.sort_values("%Match", ascending=False).reset_index(drop=True)
    df.index = df.index + 1
    df.insert(0,"Rank", df.index)

    # Why-Fit sentence via cheap chat model
    whys=[]
    for _, r in df.iterrows():
        prompt = (f'In one sentence: why does the grant titled "{r.title}" align with this mission: '
                  f'"{MISSION}"?')
        whys.append(chat(MODEL_CHAT, [{"role":"user","content": prompt}], max_tokens=60).strip())
    df["Why Fit"] = whys
    return df

# ---------- STREAMLIT UI ------------------------------------------
st.title("Final Capstone Project (Grant Matcher for CT RISE using ChatGPT Search)")
st.write("> **Mission:**", MISSION)

if st.button("🚀 Generate & Rank 10 Grants", type="primary"):
    with st.spinner("GPT-4o is searching and compiling grants…"):
        grants = get_grants()
        if len(grants) < N_GRANTS:
            st.error("GPT-4o could not find enough complete grants – try again.")
        else:
            st.session_state["tbl"] = build_table(grants)
            st.success("Done!")

if "tbl" in st.session_state:
    st.dataframe(
        st.session_state["tbl"][
            ["Rank","title","sponsor","amount","deadline","%Match","Why Fit","url","summary"]
        ],
        use_container_width=True
    )
else:
    st.caption("Press the rocket to generate your ranked grant list.")
