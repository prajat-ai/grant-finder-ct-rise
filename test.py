import os, time, json, streamlit as st, pandas as pd, openai
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

# ── CONFIG ─────────────────────────────────────────
MISSION = ("The Connecticut RISE Network empowers public high schools with "
           "data-driven strategies and personalized support to improve student outcomes "
           "and promote postsecondary success, especially for Black, Latinx, "
           "and low-income youth.")
NUM_ROWS   = 10        # final table size
EMB_MODEL  = "text-embedding-ada-002"
CHAT_MODEL = "gpt-4o-mini"      # supports `search` tool
SEARCH_TRIES = 4
LOAD_TRIES   = 3

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# ── OPENAI WRAPPERS ───────────────────────────────
def embed(txt):
    for _ in range(3):
        try:
            return openai.Embedding.create(input=txt, model=EMB_MODEL)["data"][0]["embedding"]
        except openai.error.RateLimitError:
            time.sleep(2)
    st.error("Embedding rate-limit"); st.stop()

def gpt_call(msgs, tools=None, tool_choice=None, max_tokens=800):
    return openai.ChatCompletion.create(
        model=CHAT_MODEL,
        messages=msgs,
        tools=tools or [],
        tool_choice=tool_choice,
        temperature=0.3,
        max_tokens=max_tokens,
    )

# ── 1 · fetch grants via GPT-search ────────────────
def fetch_grants() -> list[dict]:
    grants = []
    attempts = 0
    while len(grants) < NUM_ROWS and attempts < SEARCH_TRIES:
        q = f"grant opportunity high school education youth nonprofit {len(grants)+1}"
        msgs = [{"role":"user","content":f"search:{q}"}]   # triggers search tool
        res = gpt_call(msgs)
        tool_content = res.choices[0].message.tool_calls[0].function.arguments
        # tool_content contains JSON: {url, title, snippet}
        info = json.loads(tool_content)
        extract_prompt = (
            f"From the snippet below, extract JSON with keys title, sponsor, amount, "
            f"deadline, url, summary. If a field missing, write N/A.\n\nSnippet:\n{info}"
        )
        js = gpt_call([{"role":"user","content":extract_prompt}], max_tokens=400).choices[0].message.content
        try:
            g = json.loads(js)
            grants.append(g)
        except Exception:
            pass
        attempts += 1
    return grants[:NUM_ROWS]

# ── 2 · rank + annotate ───────────────────────────
def build_table(rows):
    df = pd.DataFrame(rows)
    mvec = embed(MISSION)
    df["%Match"] = (df.summary.apply(lambda s: cosine_similarity([embed(s)], [mvec])[0][0]*100)
                    .round(1))
    df = df.sort_values("%Match", ascending=False).reset_index(drop=True)
    df.index = df.index+1
    df.insert(0,"Rank",df.index)
    # Why-fit sentence
    fits=[]
    for _,r in df.iterrows():
        p = f'In one sentence: why does "{r.title}" align with "{MISSION}"?'
        fits.append(gpt_call([{"role":"user","content":p}], max_tokens=50)
                    .choices[0].message.content.strip())
    df["Why Fit"] = fits
    return df

# ── UI ─────────────────────────────────────────────
st.title("Final Capstone Project (Grant Matcher for CT RISE using ChatGPT Search)")
st.write("> **Mission:**", MISSION)

if st.button("🚀 Generate & Rank 10 Grants", type="primary"):
    with st.spinner("GPT-search is collecting grants…"):
        raw = fetch_grants()
        if not raw:
            st.error("GPT search returned no grants; try again.")
        else:
            st.session_state["tbl"] = build_table(raw)
            st.success("Done!")

if "tbl" in st.session_state:
    st.dataframe(
        st.session_state["tbl"][
            ["Rank","title","sponsor","amount","deadline","%Match","Why Fit","url","summary"]
        ],
        use_container_width=True
    )
else:
    st.caption("Click the rocket to create your grant list (powered by GPT-search).")
