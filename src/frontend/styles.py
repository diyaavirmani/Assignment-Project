"""CSS for the Streamlit 3GPP Verified RAG Assistant."""

import streamlit as st


def inject_styles() -> None:
    """Inject contained Streamlit styling for the assistant interface."""
    st.markdown(
        """
<style>
    :root {
        --page-bg: #fcfbff;
        --panel-bg: #ffffff;
        --panel-soft: #f7f2ff;
        --text-main: #111827;
        --text-muted: #7b8194;
        --border: #e8e3f2;
        --border-strong: #d8c8ff;
        --accent: #7c4ee6;
        --accent-deep: #6840d8;
        --accent-soft: #f1e8ff;
        --success: #16803c;
        --success-soft: #e7f8ed;
        --warning: #a16207;
        --warning-soft: #fff6d8;
        --danger: #b42318;
        --danger-soft: #fff0f0;
        --radius: 12px;
        --shadow: 0 18px 50px rgba(91, 33, 182, 0.08);
    }

    .stApp {
        background:
            radial-gradient(circle at 56% 7%, rgba(157, 120, 255, 0.12), transparent 28rem),
            linear-gradient(180deg, #ffffff 0%, var(--page-bg) 100%),
            var(--page-bg);
        color: var(--text-main);
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    div[data-testid="stToolbar"],
    div[data-testid="stDecoration"],
    .stDeployButton {
        display: none !important;
    }

    header[data-testid="stHeader"] {
        display: block !important;
        visibility: visible !important;
        height: 0 !important;
        background: transparent !important;
    }

    div[data-testid="collapsedControl"],
    div[data-testid="stSidebarCollapsedControl"],
    button[data-testid="stSidebarCollapseButton"],
    section[data-testid="stSidebar"] button[title="Close sidebar"],
    section[data-testid="stSidebar"] button[aria-label="Close sidebar"] {
        display: none !important;
    }

    .block-container {
        max-width: 1280px;
        padding-top: 1.45rem;
        padding-bottom: 2.25rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }

    section[data-testid="stSidebar"] {
        display: block !important;
        visibility: visible !important;
        opacity: 1 !important;
        min-width: 21.5rem !important;
        width: 21.5rem !important;
        transform: translateX(0) !important;
        background: linear-gradient(180deg, rgba(255,255,255,0.98) 0%, #fbf9ff 100%);
        border-right: 1px solid var(--border);
        box-shadow: 16px 0 44px rgba(91, 33, 182, 0.035);
    }

    section[data-testid="stSidebar"] .block-container {
        padding: 1.8rem 1.25rem 1.35rem;
    }

    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 0.82rem;
        margin-bottom: 1.28rem;
    }

    .brand-mark {
        width: 2.42rem;
        height: 2.42rem;
        border-radius: 50%;
        background:
            radial-gradient(circle at 35% 35%, #ffffff 0 14%, transparent 15%),
            radial-gradient(circle at 56% 56%, rgba(255,255,255,0.88) 0 24%, transparent 25%),
            linear-gradient(135deg, #efe8ff, #b996ff 52%, #ffffff);
        border: 1px solid #dac8ff;
        box-shadow: 0 10px 28px rgba(124, 58, 237, 0.20), inset 0 0 18px rgba(255,255,255,0.75);
    }

    .brand-title {
        font-weight: 800;
        font-size: 1.08rem;
        color: var(--text-main);
        line-height: 1.15;
    }

    .brand-subtitle,
    .small-muted {
        color: var(--text-muted);
        font-size: 0.84rem;
    }

    .verified-chip,
    .badge {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        border-radius: 999px;
        border: 1px solid var(--border);
        padding: 0.26rem 0.62rem;
        background: var(--panel-soft);
        color: var(--text-main);
        font-size: 0.74rem;
        font-weight: 760;
        text-transform: uppercase;
    }

    .verified-chip.ok,
    .badge.ok {
        color: var(--success);
        border-color: #b7e7c7;
        background: var(--success-soft);
    }

    .verified-chip.danger,
    .badge.danger {
        color: var(--danger);
        border-color: #ffd0d0;
        background: var(--danger-soft);
    }

    .badge.warn {
        color: var(--warning);
        border-color: #f3d46f;
        background: var(--warning-soft);
    }

    .hero {
        text-align: center;
        padding: 2.1rem 1rem 1.55rem;
        margin: 0 auto;
        max-width: 820px;
    }

    .hero-orb-wrap {
        width: 8.6rem;
        height: 8.6rem;
        position: relative;
        margin: 0 auto 0.55rem;
    }

    .hero-orb {
        position: absolute;
        inset: 1.4rem;
        border-radius: 50%;
        background:
            radial-gradient(circle at 42% 36%, #ffffff 0 18%, rgba(255,255,255,0.52) 32%, transparent 54%),
            radial-gradient(circle at 60% 62%, rgba(124, 78, 230, 0.42), transparent 58%),
            linear-gradient(135deg, #f7f1ff 0%, #bd9aff 62%, #ffffff 100%);
        border: 1px solid rgba(173, 137, 248, 0.44);
        box-shadow:
            0 24px 74px rgba(124, 78, 230, 0.24),
            inset 0 0 26px rgba(255,255,255,0.86);
    }

    .hero-orb::before,
    .hero-orb::after {
        content: "";
        position: absolute;
        inset: -1.08rem;
        border-radius: 50%;
        border: 0.8rem solid rgba(160, 122, 239, 0.24);
        transform: rotate(-28deg) skew(6deg);
        filter: blur(0.2px);
    }

    .hero-orb::after {
        inset: -0.55rem;
        border-width: 0.55rem;
        border-color: rgba(124, 78, 230, 0.16);
        transform: rotate(24deg) skew(-8deg);
    }

    .sparkle {
        position: absolute;
        width: 0.24rem;
        height: 0.24rem;
        border-radius: 50%;
        background: var(--accent);
        box-shadow: 0 0 10px rgba(124, 78, 230, 0.6);
    }

    .sparkle.s1 { right: 1.1rem; top: 2.1rem; }
    .sparkle.s2 { left: 0.95rem; bottom: 2.4rem; }
    .sparkle.s3 { right: 2.1rem; bottom: 1.65rem; width: 0.16rem; height: 0.16rem; }

    .hero h1 {
        font-size: clamp(2.8rem, 4.8vw, 4rem);
        line-height: 0.98;
        margin: 0;
        color: var(--accent);
        font-weight: 820;
        letter-spacing: 0;
        background: linear-gradient(180deg, #b58bff 0%, #7c4ee6 100%);
        -webkit-background-clip: text;
        color: transparent;
    }

    .hero h2 {
        margin: 0.44rem 0 0.48rem;
        color: var(--text-main);
        font-size: clamp(1.5rem, 2.6vw, 2.05rem);
        font-weight: 800;
        letter-spacing: 0;
    }

    .hero p {
        margin: 0;
        color: var(--text-muted);
        font-size: 1rem;
    }

    .composer-intro {
        max-width: 760px;
        margin: 0 auto 0.62rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        color: var(--text-muted);
        font-size: 0.9rem;
    }

    .composer-intro strong {
        color: var(--accent-deep);
        font-weight: 760;
    }

    div[data-testid="stForm"] {
        max-width: 760px;
        margin: 0 auto 1rem;
        background: rgba(255, 255, 255, 0.94);
        border: 1px solid #eadfff;
        border-radius: 16px;
        padding: 0.95rem 1rem 0.75rem;
        box-shadow: 0 18px 58px rgba(91, 33, 182, 0.095);
    }

    div[data-testid="stForm"] textarea {
        min-height: 7.7rem !important;
        border: 0 !important;
        box-shadow: none !important;
        background: transparent !important;
        color: var(--text-main) !important;
        font-size: 1.02rem !important;
        padding: 0.15rem 0 !important;
        resize: none;
    }

    div[data-testid="stForm"] div[data-baseweb="textarea"] {
        border: 0 !important;
        box-shadow: none !important;
        background: transparent !important;
    }

    div[data-testid="stForm"] textarea::placeholder {
        color: #98a0b3;
    }

    div[data-testid="stForm"] [data-testid="stTextArea"] {
        border-bottom: 1px solid #efe7ff;
        margin-bottom: 0.72rem;
    }

    .composer-mode-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.48rem;
        min-height: 2.45rem;
        padding: 0 0.82rem;
        border: 1px solid var(--border-strong);
        border-radius: 10px;
        background: #fbf9ff;
        color: var(--accent-deep);
        font-weight: 700;
        font-size: 0.9rem;
    }

    .composer-helper {
        margin: 0.6rem -1rem -0.75rem;
        padding: 0.72rem 1rem;
        border-top: 1px solid #efe7ff;
        background: linear-gradient(180deg, rgba(251,249,255,0.86), rgba(255,255,255,0.9));
        display: flex;
        align-items: center;
        gap: 0.5rem;
        color: var(--text-muted);
        font-size: 0.84rem;
    }

    .capability-card,
    .metric-card,
    .source-card {
        background: rgba(255, 255, 255, 0.92);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        box-shadow: 0 8px 22px rgba(91, 33, 182, 0.045);
    }

    .capability-card {
        display: flex;
        gap: 1rem;
        padding: 1.35rem 1.15rem;
        min-height: 8.75rem;
        margin-top: 1rem;
    }

    .capability-icon {
        flex: 0 0 auto;
        width: 2.7rem;
        height: 2.7rem;
        border-radius: 50%;
        display: grid;
        place-items: center;
        color: var(--accent);
        background: linear-gradient(135deg, #f5efff, #ffffff);
        border: 1px solid #eadfff;
        font-size: 1.32rem;
        font-weight: 900;
    }

    .capability-title {
        color: var(--text-main);
        font-size: 1rem;
        font-weight: 760;
        margin-bottom: 0.4rem;
    }

    .capability-copy {
        color: var(--text-muted);
        font-size: 0.9rem;
        line-height: 1.45;
    }

    .section-label {
        margin: 1.15rem 0 0.45rem;
        color: var(--text-muted);
        font-size: 0.72rem;
        font-weight: 780;
        letter-spacing: 0;
        text-transform: uppercase;
    }

    .status-panel {
        max-width: 780px;
        margin: 2rem auto 0.65rem;
        display: grid;
        grid-template-columns: minmax(7.5rem, 0.55fr) 2.8fr;
        align-items: center;
        gap: 0.85rem;
        border: 1px solid var(--border);
        border-radius: 14px;
        background: rgba(255, 255, 255, 0.84);
        padding: 0.74rem 0.88rem;
        box-shadow: 0 10px 28px rgba(91, 33, 182, 0.035);
    }

    .status-panel .section-label {
        margin: 0;
    }

    .status-items {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.65rem;
    }

    .status-item {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.38rem;
        color: var(--text-main);
        font-size: 0.78rem;
        font-weight: 700;
        white-space: nowrap;
        border: 1px solid var(--border);
        border-radius: 10px;
        background: #ffffff;
        padding: 0.55rem 0.6rem;
    }

    .dot {
        width: 0.38rem;
        height: 0.38rem;
        border-radius: 999px;
        display: inline-block;
        background: #d1d5db;
    }

    .dot.ok {
        background: #22c55e;
        box-shadow: 0 0 0 3px rgba(34, 197, 94, 0.12);
    }

    .dot.danger {
        background: #ef4444;
        box-shadow: 0 0 0 3px rgba(239, 68, 68, 0.12);
    }

    .status-glyph {
        width: 1.05rem;
        height: 1.05rem;
        border: 1px solid #ece4ff;
        border-radius: 0.32rem;
        background: #fbf9ff;
    }

    .message-meta {
        margin-top: 0.7rem;
        display: flex;
        flex-wrap: wrap;
        gap: 0.4rem;
    }

    .answer-state {
        display: flex;
        gap: 0.6rem;
        align-items: center;
        flex-wrap: wrap;
        border-radius: var(--radius);
        border: 1px solid var(--border);
        padding: 0.58rem 0.7rem;
        margin-top: 0.75rem;
        margin-bottom: 0.72rem;
        font-size: 0.84rem;
    }

    .answer-state strong {
        font-size: 0.76rem;
        letter-spacing: 0;
    }

    .answer-state.ok {
        background: var(--success-soft);
        border-color: #b7e7c7;
        color: var(--success);
    }

    .answer-state.blocked {
        background: var(--warning-soft);
        border-color: #f3d46f;
        color: var(--warning);
    }

    .answer-state.failed {
        background: var(--danger-soft);
        border-color: #ffd0d0;
        color: var(--danger);
    }

    .source-card {
        padding: 0.78rem 0.85rem;
        margin-bottom: 0.65rem;
    }

    .source-id {
        font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
        color: var(--accent);
        font-weight: 820;
        font-size: 0.84rem;
    }

    .source-title {
        color: var(--text-main);
        font-weight: 720;
        margin: 0.16rem 0;
    }

    .source-subtitle {
        color: var(--text-muted);
        font-size: 0.82rem;
    }

    .source-metrics {
        display: flex;
        flex-wrap: wrap;
        gap: 0.42rem;
        margin-top: 0.55rem;
    }

    .source-metrics span {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        border: 1px solid var(--border);
        border-radius: 999px;
        background: #fbf9ff;
        color: var(--text-muted);
        font-size: 0.76rem;
        padding: 0.24rem 0.52rem;
    }

    .source-metrics b {
        color: var(--text-main);
        font-weight: 760;
    }

    .metric-card {
        padding: 0.65rem;
        min-height: 4.25rem;
        margin-bottom: 0.5rem;
    }

    .metric-label {
        color: var(--text-muted);
        font-size: 0.69rem;
        text-transform: uppercase;
        font-weight: 760;
    }

    .metric-value {
        font-size: 1.06rem;
        font-weight: 780;
        color: var(--text-main);
        margin-top: 0.15rem;
    }

    .corpus-count {
        color: var(--text-muted);
        font-size: 0.85rem;
        margin-bottom: 0.45rem;
    }

    .indexed-spec-card {
        display: flex;
        align-items: center;
        gap: 0.72rem;
        border: 1px solid var(--border);
        border-radius: var(--radius);
        background: rgba(255, 255, 255, 0.86);
        padding: 0.62rem 0.68rem;
        margin: 0.45rem 0;
        box-shadow: 0 8px 18px rgba(91, 33, 182, 0.04);
    }

    .indexed-spec-icon {
        width: 2.2rem;
        height: 2.2rem;
        flex: 0 0 auto;
        border-radius: 0.62rem;
        display: grid;
        place-items: center;
        color: var(--accent);
        background: #f3ecff;
        font-weight: 900;
    }

    .indexed-spec-body {
        min-width: 0;
        flex: 1;
    }

    .indexed-spec-head {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 0.5rem;
        color: var(--text-main);
        font-size: 0.86rem;
    }

    .indexed-spec-head span {
        display: inline-flex;
        align-items: center;
        gap: 0.28rem;
        color: var(--success);
        font-size: 0.68rem;
        font-weight: 780;
    }

    .indexed-spec-head span i {
        width: 0.36rem;
        height: 0.36rem;
        border-radius: 50%;
        background: #22c55e;
    }

    .indexed-spec-title {
        color: var(--text-muted);
        font-size: 0.78rem;
        line-height: 1.35;
        margin-top: 0.28rem;
    }

    .catalog-row {
        padding: 0.5rem 0;
        border-bottom: 1px solid #f0e8ff;
        color: var(--text-main);
        font-size: 0.84rem;
    }

    .catalog-main {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 0.5rem;
        font-weight: 720;
    }

    .catalog-main span {
        color: var(--text-muted);
        font-size: 0.68rem;
        font-weight: 720;
        text-align: right;
    }

    .catalog-title {
        color: var(--text-muted);
        font-size: 0.75rem;
        line-height: 1.32;
        margin-top: 0.16rem;
    }

    .catalog-row small {
        color: var(--text-muted);
        font-size: 0.72rem;
    }

    .catalog-row.active .catalog-main {
        color: var(--accent-deep);
    }

    .offline-panel {
        border: 1px solid #ffd0d0;
        background: var(--danger-soft);
        border-radius: var(--radius);
        padding: 0.82rem;
        color: var(--danger);
        font-size: 0.86rem;
    }

    div[data-testid="stChatMessage"] {
        background: rgba(255, 255, 255, 0.82);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 0.72rem;
        margin-bottom: 0.75rem;
        box-shadow: 0 8px 22px rgba(91, 33, 182, 0.04);
    }

    .stButton > button,
    .stDownloadButton > button {
        border-radius: var(--radius);
        border: 1px solid var(--border);
        background: #ffffff;
        color: var(--text-main);
        font-weight: 660;
        min-height: 2.55rem;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        border-color: var(--border-strong);
        color: var(--accent-deep);
        background: var(--accent-soft);
    }

    button[kind="primary"],
    button[data-testid="stBaseButton-primary"],
    .stButton > button[kind="primary"],
    .stButton > button[data-testid="stBaseButton-primary"],
    .stForm button[kind="primary"],
    .stForm button[data-testid="stBaseButton-primary"] {
        border: 0 !important;
        color: #ffffff !important;
        background: linear-gradient(135deg, #8b5cf6, #6d46d9) !important;
        box-shadow: 0 12px 24px rgba(124, 78, 230, 0.22) !important;
    }

    div[data-testid="stChatInput"] {
        max-width: 820px;
        margin: 0 auto;
    }

    @media (max-width: 900px) {
        .hero {
            padding-top: 2.2rem;
        }

        .status-panel {
            grid-template-columns: 1fr;
        }

        .status-items {
            grid-template-columns: 1fr 1fr;
        }

        .status-item,
        .composer-intro {
            white-space: normal;
        }

        .capability-card {
            min-height: auto;
        }
    }
</style>
        """,
        unsafe_allow_html=True,
    )
