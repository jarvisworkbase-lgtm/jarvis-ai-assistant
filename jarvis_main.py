"""
JARVIS AI Assistant - jarvis_main.py
======================================
A sleek, Iron Man-inspired AI assistant with web search, voice output,
conversation memory, and a rich Streamlit UI.

Requirements:
    pip install streamlit openai tavily-python pyttsx3

Run:
    streamlit run jarvis_main.py
"""

# ─────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────
import threading
import datetime
import json
import time
from typing import Optional

import streamlit as st
from openai import OpenAI, APIConnectionError, APIStatusError
from tavily import TavilyClient
import pyttsx3


# ─────────────────────────────────────────────────────────────
# CONFIGURATION  —  edit these values to match your setup
# ─────────────────────────────────────────────────────────────
class Config:
    """Central configuration for the JARVIS assistant."""

    # Ollama / LLM settings
    OLLAMA_BASE_URL: str = "http://localhost:11434/v1"
    OLLAMA_API_KEY: str  = "ollama"           # Ollama does not validate this
    DEFAULT_MODEL: str   = "llama3"

    # Tavily web-search settings
    TAVILY_API_KEY: str  = "tvly-dev-REPLACE_WITH_YOUR_KEY"
    MAX_SEARCH_RESULTS: int = 5
    MAX_SNIPPET_LENGTH: int = 350             # chars per search snippet

    # LLM generation settings
    DEFAULT_TEMPERATURE: float = 0.7
    MAX_HISTORY_TURNS: int     = 10           # how many past turns to send to LLM
    MAX_RESPONSE_TOKENS: int   = 1024

    # Voice settings
    VOICE_RATE: int     = 175
    VOICE_VOLUME: float = 1.0
    VOICE_DRIVER: str   = "sapi5"            # Windows: "sapi5" | macOS: "nsss" | Linux: "espeak"

    # System persona
    SYSTEM_PROMPT: str = (
        "You are JARVIS (Just A Rather Very Intelligent System), "
        "a sharp, sophisticated, and concise AI assistant. "
        "You speak with calm confidence, a dry wit, and surgical precision. "
        "Use the provided web context to answer accurately. "
        "Never mention that you were given external context — answer naturally. "
        "If unsure, say so clearly rather than guessing."
    )


# ─────────────────────────────────────────────────────────────
# VOICE ENGINE
# ─────────────────────────────────────────────────────────────
class VoiceEngine:
    """Handles text-to-speech in a background thread so the UI never blocks."""

    @staticmethod
    def speak(text: str) -> None:
        """Speak *text* asynchronously in a daemon thread."""
        thread = threading.Thread(
            target=VoiceEngine._speak_blocking,
            args=(text,),
            daemon=True,
        )
        thread.start()

    @staticmethod
    def _speak_blocking(text: str) -> None:
        """Blocking TTS call — runs inside a background thread."""
        try:
            engine = pyttsx3.init(driverName=Config.VOICE_DRIVER)
            engine.setProperty("rate",   Config.VOICE_RATE)
            engine.setProperty("volume", Config.VOICE_VOLUME)

            # Pick the first English voice available
            for voice in engine.getProperty("voices"):
                if "english" in voice.name.lower():
                    engine.setProperty("voice", voice.id)
                    break

            engine.say(text)
            engine.runAndWait()
            engine.stop()
        except Exception as exc:                    # noqa: BLE001
            # Voice failure is non-fatal — log quietly
            print(f"[VoiceEngine] TTS error: {exc}")


# ─────────────────────────────────────────────────────────────
# WEB SEARCH
# ─────────────────────────────────────────────────────────────
class WebSearch:
    """Wraps the Tavily client with clean error handling."""

    def __init__(self) -> None:
        self._client = TavilyClient(api_key=Config.TAVILY_API_KEY)

    def search(self, query: str) -> tuple[str, list[dict]]:
        """
        Search the web for *query*.

        Returns
        -------
        context : str
            Formatted snippet string to inject into the LLM prompt.
        raw_results : list[dict]
            Raw Tavily result dicts for optional display in the UI.
        """
        try:
            response = self._client.search(
                query=query,
                max_results=Config.MAX_SEARCH_RESULTS,
            )
            results = response.get("results", [])
        except Exception as exc:                    # noqa: BLE001
            print(f"[WebSearch] Search failed: {exc}")
            return "", []

        snippets = []
        for r in results:
            title   = r.get("title",   "No title")
            content = r.get("content", "")[:Config.MAX_SNIPPET_LENGTH]
            url     = r.get("url",     "")
            snippets.append(f"• {title}\n  {content}\n  Source: {url}")

        context = "\n\n".join(snippets)
        return context, results


# ─────────────────────────────────────────────────────────────
# LLM CLIENT
# ─────────────────────────────────────────────────────────────
class LLMClient:
    """Wraps the Ollama-compatible OpenAI client."""

    def __init__(self) -> None:
        self._client = OpenAI(
            base_url=Config.OLLAMA_BASE_URL,
            api_key=Config.OLLAMA_API_KEY,
        )

    def chat(
        self,
        user_message: str,
        history: list[dict],
        model: str,
        temperature: float,
    ) -> tuple[str, float]:
        """
        Send *user_message* (with *history*) to the LLM.

        Returns
        -------
        answer : str
            The model's reply.
        elapsed : float
            Wall-clock seconds the inference took.
        """
        messages = [{"role": "system", "content": Config.SYSTEM_PROMPT}]

        # Append limited conversation history for context
        messages.extend(history[-(Config.MAX_HISTORY_TURNS * 2):])
        messages.append({"role": "user", "content": user_message})

        start = time.time()
        try:
            response = self._client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=Config.MAX_RESPONSE_TOKENS,
                stream=False,
            )
            answer = response.choices[0].message.content.strip()
        except APIConnectionError:
            answer = (
                "⚠ Cannot reach Ollama. "
                "Please ensure Ollama is running on "
                f"`{Config.OLLAMA_BASE_URL}`."
            )
        except APIStatusError as exc:
            answer = f"⚠ API error {exc.status_code}: {exc.message}"
        except Exception as exc:                    # noqa: BLE001
            answer = f"⚠ Unexpected error: {exc}"

        elapsed = time.time() - start
        return answer, elapsed


# ─────────────────────────────────────────────────────────────
# JARVIS CORE  —  orchestrates search → prompt → LLM → voice
# ─────────────────────────────────────────────────────────────
class JARVIS:
    """Main assistant class that wires together all subsystems."""

    def __init__(self) -> None:
        self._search = WebSearch()
        self._llm    = LLMClient()
        self._voice  = VoiceEngine()

    def respond(
        self,
        user_query: str,
        history: list[dict],
        model: str,
        temperature: float,
        voice_enabled: bool,
        web_search_enabled: bool,
    ) -> dict:
        """
        Full pipeline: search → build prompt → LLM → (optional) TTS.

        Returns a dict with keys:
            answer, elapsed, sources, timestamp
        """
        # ── 1. Optional web search ────────────────────────────
        context, sources = "", []
        if web_search_enabled:
            context, sources = self._search.search(user_query)

        # ── 2. Compose the user message ───────────────────────
        if context:
            user_message = (
                f"Web context (fresh search results):\n{context}\n\n"
                f"User question: {user_query}"
            )
        else:
            user_message = user_query

        # ── 3. LLM inference ──────────────────────────────────
        answer, elapsed = self._llm.chat(
            user_message=user_message,
            history=history,
            model=model,
            temperature=temperature,
        )

        # ── 4. Optional voice output ──────────────────────────
        if voice_enabled and not answer.startswith("⚠"):
            self._voice.speak(answer)

        return {
            "answer":    answer,
            "elapsed":   elapsed,
            "sources":   sources,
            "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
        }


# ─────────────────────────────────────────────────────────────
# SESSION-STATE HELPERS
# ─────────────────────────────────────────────────────────────
def init_session_state() -> None:
    """Initialise all Streamlit session-state keys on first run."""
    defaults = {
        "history":          [],     # list of (query, result_dict)
        "llm_history":      [],     # OpenAI-format message list for the LLM
        "voice_enabled":    True,
        "web_search_enabled": True,
        "model":            Config.DEFAULT_MODEL,
        "temperature":      Config.DEFAULT_TEMPERATURE,
        "show_sources":     False,
        "jarvis":           JARVIS(),
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def export_conversation() -> str:
    """Serialise the conversation history to a JSON string."""
    export_data = []
    for query, result in st.session_state.history:
        export_data.append({
            "timestamp": result["timestamp"],
            "query":     query,
            "answer":    result["answer"],
            "sources":   [s.get("url", "") for s in result.get("sources", [])],
        })
    return json.dumps(export_data, indent=2, ensure_ascii=False)


# ─────────────────────────────────────────────────────────────
# CSS  —  Iron Man HUD aesthetic
# ─────────────────────────────────────────────────────────────
GLOBAL_CSS = """
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700;900&family=Share+Tech+Mono&family=Exo+2:wght@300;400;600&display=swap');

/* ── Base / reset ── */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stHeader"],
[data-testid="stMain"] {
    background: #000 !important;
    color: #c8e8ff !important;
}
section[data-testid="stSidebar"] {
    background: #02060f !important;
    border-right: 1px solid #1a6aff22 !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"]  { display: none; }

/* ── Global typography ── */
*, p, div, label, span {
    font-family: 'Share Tech Mono', monospace !important;
    color: #c8e8ff;
}

/* ── Inputs ── */
textarea,
input[type="text"],
.stTextInput input {
    background:  #050d1f !important;
    border:      1px solid #1a6aff44 !important;
    border-radius: 6px !important;
    color:       #c8e8ff !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size:   15px !important;
    caret-color: #1a6aff !important;
    transition:  border-color .2s, box-shadow .2s !important;
}
textarea:focus,
.stTextInput input:focus {
    border-color: #1a6aff !important;
    box-shadow:   0 0 14px #1a6aff44 !important;
    outline: none !important;
}

/* ── Sliders ── */
.stSlider [data-baseweb="slider"] > div:nth-child(2) > div {
    background: #1a6aff !important;
}

/* ── Select box ── */
[data-baseweb="select"] > div {
    background: #050d1f !important;
    border: 1px solid #1a6aff33 !important;
    border-radius: 6px !important;
}
[data-baseweb="select"] span { color: #c8e8ff !important; }

/* ── Primary / engage button ── */
.stButton > button {
    background:    transparent !important;
    border:        1px solid #1a6aff !important;
    border-radius: 6px !important;
    color:         #1a6aff !important;
    font-family:   'Orbitron', sans-serif !important;
    font-size:     13px !important;
    letter-spacing: 2px !important;
    padding:       8px 28px !important;
    transition:    all .25s ease !important;
}
.stButton > button:hover {
    background:  #1a6aff18 !important;
    box-shadow:  0 0 20px #1a6affbb !important;
    color:       #ffffff !important;
    transform:   translateY(-1px) !important;
}
.stButton > button:active {
    transform: translateY(0) !important;
    box-shadow: 0 0 8px #1a6aff66 !important;
}

/* ── Danger / clear button ── */
button[kind="secondary"] {
    border-color: #ff3b3b66 !important;
    color:        #ff6b6b !important;
}
button[kind="secondary"]:hover {
    background:  #ff3b3b18 !important;
    box-shadow:  0 0 14px #ff3b3b88 !important;
    color:       #ffaaaa !important;
}

/* ── Toggle / checkbox ── */
.stCheckbox span { color: #c8e8ff !important; }
[data-baseweb="checkbox"] > div > div {
    border-color: #1a6aff88 !important;
}
[data-baseweb="checkbox"] [aria-checked="true"] > div {
    background: #1a6aff !important;
}

/* ── Response card ── */
.response-box {
    background:    #04091a;
    border:        1px solid #1a6aff33;
    border-left:   3px solid #1a6aff;
    border-radius: 8px;
    padding:       18px 22px;
    margin-top:    10px;
    font-family:   'Share Tech Mono', monospace;
    font-size:     14px;
    color:         #c8e8ff;
    line-height:   1.8;
    box-shadow:    0 0 20px #1a6aff11 inset,
                   0 2px 12px #00000066;
    white-space:   pre-wrap;
}

/* ── Meta row (timestamp, elapsed, char-count) ── */
.meta-row {
    display:         flex;
    gap:             18px;
    flex-wrap:       wrap;
    margin-top:      6px;
    font-size:       11px;
    color:           #1a6aff88 !important;
    letter-spacing:  1px;
}
.meta-row span { color: #1a6aff88 !important; }

/* ── Source pill ── */
.source-pill {
    display:         inline-block;
    background:      #0a1628;
    border:          1px solid #1a6aff33;
    border-radius:   4px;
    padding:         2px 10px;
    margin:          3px 4px 3px 0;
    font-size:       11px;
    color:           #6aabff !important;
    text-decoration: none;
    transition:      background .2s;
}
.source-pill:hover { background: #1a6aff22; }

/* ── Section label (query echo) ── */
.query-label {
    font-size:      11px;
    letter-spacing: 2px;
    color:          #1a6aff88 !important;
    margin-bottom:  2px;
}

/* ── Divider ── */
hr { border-color: #1a6aff18 !important; }

/* ── Orb / pulse ── */
.orb-wrapper {
    display:         flex;
    justify-content: center;
    align-items:     center;
    height:          180px;
    margin:          8px 0 4px;
}
.orb {
    width:         100px;
    height:        100px;
    border-radius: 50%;
    background:    radial-gradient(circle at 34% 34%,
                        #72cfff 0%,
                        #1a6aff 35%,
                        #0033cc 65%,
                        #000820 100%);
    box-shadow:
        0 0 28px  #1a6aff,
        0 0 65px  #1a6affaa,
        0 0 120px #1a6aff55,
        0 0 180px #1a6aff22;
    animation: pulse 2.8s ease-in-out infinite;
    position:  relative;
}
.orb::after {
    content:       '';
    position:      absolute;
    inset:         11px;
    border-radius: 50%;
    background:    radial-gradient(circle at 30% 30%,
                        #ffffff44 0%,
                        transparent 60%);
}
@keyframes pulse {
    0%   { transform: scale(1);    box-shadow: 0 0 28px #1a6aff, 0 0 65px #1a6affaa, 0 0 120px #1a6aff55; }
    50%  { transform: scale(1.09); box-shadow: 0 0 44px #1a6aff, 0 0 100px #1a6affcc, 0 0 170px #1a6aff88; }
    100% { transform: scale(1);    box-shadow: 0 0 28px #1a6aff, 0 0  65px #1a6affaa, 0 0 120px #1a6aff55; }
}

/* ── Title ── */
.jarvis-title {
    text-align:     center;
    font-family:    'Orbitron', sans-serif !important;
    font-size:      40px !important;
    font-weight:    900 !important;
    letter-spacing: 12px !important;
    color:          #1a6aff !important;
    text-shadow:    0 0 18px #1a6affcc, 0 0 48px #1a6aff55;
    margin-bottom:  2px;
}
.jarvis-sub {
    text-align:     center;
    font-size:      10px !important;
    letter-spacing: 5px !important;
    color:          #1a6aff77 !important;
    margin-bottom:  0;
}

/* ── Sidebar section headers ── */
.sidebar-section {
    font-family:    'Orbitron', sans-serif !important;
    font-size:      10px !important;
    letter-spacing: 3px !important;
    color:          #1a6aff88 !important;
    margin:         18px 0 6px;
    border-bottom:  1px solid #1a6aff22;
    padding-bottom: 4px;
}
</style>
"""


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
def render_sidebar() -> None:
    """Render settings panel in the left sidebar."""
    with st.sidebar:
        st.markdown('<p class="jarvis-title" style="font-size:22px!important;letter-spacing:6px!important">⚙ SYSTEMS</p>', unsafe_allow_html=True)
        st.markdown("<hr>", unsafe_allow_html=True)

        # ── Model ─────────────────────────────────────────────
        st.markdown('<p class="sidebar-section">▸ MODEL</p>', unsafe_allow_html=True)
        model_input = st.text_input(
            "Ollama model name",
            value=st.session_state.model,
            label_visibility="collapsed",
            placeholder="e.g. llama3, mistral, phi3",
        )
        if model_input.strip():
            st.session_state.model = model_input.strip()

        # ── Temperature ───────────────────────────────────────
        st.markdown('<p class="sidebar-section">▸ TEMPERATURE</p>', unsafe_allow_html=True)
        st.session_state.temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=float(st.session_state.temperature),
            step=0.05,
            label_visibility="collapsed",
            help="Higher = more creative. Lower = more focused.",
        )
        st.caption(f"Current: {st.session_state.temperature:.2f}")

        # ── Toggles ───────────────────────────────────────────
        st.markdown('<p class="sidebar-section">▸ FEATURES</p>', unsafe_allow_html=True)
        st.session_state.voice_enabled = st.checkbox(
            "🔊  Voice output (TTS)",
            value=st.session_state.voice_enabled,
        )
        st.session_state.web_search_enabled = st.checkbox(
            "🌐  Live web search",
            value=st.session_state.web_search_enabled,
        )
        st.session_state.show_sources = st.checkbox(
            "🔗  Show sources",
            value=st.session_state.show_sources,
        )

        # ── Stats ─────────────────────────────────────────────
        st.markdown('<p class="sidebar-section">▸ SESSION STATS</p>', unsafe_allow_html=True)
        n = len(st.session_state.history)
        st.caption(f"Turns this session: **{n}**")
        if n:
            elapsed_vals = [r["elapsed"] for _, r in st.session_state.history if "elapsed" in r]
            if elapsed_vals:
                avg = sum(elapsed_vals) / len(elapsed_vals)
                st.caption(f"Avg response time: **{avg:.1f}s**")

        # ── Actions ───────────────────────────────────────────
        st.markdown('<p class="sidebar-section">▸ ACTIONS</p>', unsafe_allow_html=True)

        if n:
            export_json = export_conversation()
            st.download_button(
                label="📥  Export chat (JSON)",
                data=export_json,
                file_name=f"jarvis_session_{datetime.date.today()}.json",
                mime="application/json",
                use_container_width=True,
            )

        if st.button("🗑  Clear history", use_container_width=True):
            st.session_state.history    = []
            st.session_state.llm_history = []
            st.rerun()

        # ── Footer ────────────────────────────────────────────
        st.markdown("<hr>", unsafe_allow_html=True)
        st.caption("JARVIS v2.0  •  Powered by Ollama + Tavily")


# ─────────────────────────────────────────────────────────────
# CONVERSATION RENDERER
# ─────────────────────────────────────────────────────────────
def render_history() -> None:
    """Render all past turns newest-first."""
    for query, result in st.session_state.history:
        answer    = result["answer"]
        elapsed   = result.get("elapsed",   0.0)
        timestamp = result.get("timestamp", "")
        sources   = result.get("sources",   [])

        # Query label
        st.markdown(
            f'<p class="query-label">▸ {query}</p>',
            unsafe_allow_html=True,
        )

        # Answer card
        st.markdown(
            f'<div class="response-box">{answer}</div>',
            unsafe_allow_html=True,
        )

        # Meta row
        char_count = len(answer)
        st.markdown(
            f"""
            <div class="meta-row">
                <span>🕐 {timestamp}</span>
                <span>⚡ {elapsed:.2f}s</span>
                <span>📝 {char_count} chars</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Optional sources
        if st.session_state.show_sources and sources:
            pills_html = " ".join(
                f'<a class="source-pill" href="{s.get("url","#")}" '
                f'target="_blank" rel="noopener">🔗 {s.get("title","Source")[:40]}</a>'
                for s in sources
            )
            st.markdown(
                f'<div style="margin-top:8px">{pills_html}</div>',
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────
def main() -> None:
    # ── Page config ───────────────────────────────────────────
    st.set_page_config(
        page_title="JARVIS",
        page_icon="🔵",
        layout="centered",
        initial_sidebar_state="expanded",
    )

    # ── Inject CSS ────────────────────────────────────────────
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

    # ── Session state ─────────────────────────────────────────
    init_session_state()

    # ── Sidebar ───────────────────────────────────────────────
    render_sidebar()

    # ── Header ────────────────────────────────────────────────
    st.markdown('<p class="jarvis-title">J.A.R.V.I.S</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="jarvis-sub">JUST A RATHER VERY INTELLIGENT SYSTEM</p>',
        unsafe_allow_html=True,
    )

    # ── Animated orb ─────────────────────────────────────────
    st.markdown(
        '<div class="orb-wrapper"><div class="orb"></div></div>',
        unsafe_allow_html=True,
    )

    # ── Status indicators ────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    col1.caption(
        f"{'🟢' if st.session_state.voice_enabled else '🔴'} Voice"
    )
    col2.caption(
        f"{'🟢' if st.session_state.web_search_enabled else '🔴'} Web Search"
    )
    col3.caption(f"🤖 {st.session_state.model}")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Query form ────────────────────────────────────────────
    with st.form("query_form", clear_on_submit=True):
        user_input = st.text_input(
            "QUERY",
            placeholder="Ask me anything…",
            label_visibility="collapsed",
        )
        col_btn, col_hint = st.columns([1, 3])
        with col_btn:
            submitted = st.form_submit_button("▶  ENGAGE", use_container_width=True)
        with col_hint:
            st.caption("Press Enter or click ENGAGE to send your query.")

    # ── Handle submission ─────────────────────────────────────
    if submitted and user_input.strip():
        query = user_input.strip()

        with st.spinner("🔵  Processing query…"):
            result = st.session_state.jarvis.respond(
                user_query=query,
                history=st.session_state.llm_history,
                model=st.session_state.model,
                temperature=st.session_state.temperature,
                voice_enabled=st.session_state.voice_enabled,
                web_search_enabled=st.session_state.web_search_enabled,
            )

        # Save to display history (newest first)
        st.session_state.history.insert(0, (query, result))

        # Update LLM conversation memory
        st.session_state.llm_history.append({"role": "user",      "content": query})
        st.session_state.llm_history.append({"role": "assistant",  "content": result["answer"]})

        st.rerun()

    elif submitted:
        st.warning("Please enter a query before engaging.")

    # ── Render conversation history ───────────────────────────
    if st.session_state.history:
        render_history()
    else:
        st.markdown(
            """
            <div style="text-align:center;margin-top:40px;opacity:.4">
                <p style="font-size:13px;letter-spacing:3px">AWAITING YOUR COMMAND, SIR.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
