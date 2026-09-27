"""Editorial Design System Tokens and Styling (Stitch Specification).

Color Palette:
- Primary Background: Soft sky / slate blue (#9BB8D4 / #A8C4E0)
- Accent & Typography: Deep espresso brown (#111111)
- Container Surface: Warm cream (#FFFFFF / #EAEAEA)
- Hairline Borders: Warm sand (#E2E8F0 / #CCCCCC)
"""

EDITORIAL_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400..700;1,6..72,400..700&family=Plus+Jakarta+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* Global Base Canvas */
    .stApp {
        background-color: #FAFAFA;
        background-image: none;
        color: #111111;
        font-family: 'Newsreader', Georgia, serif;
    }

    /* Editorial Header Bar */
    .editorial-header {
        background: #FFFFFF;
        border: 1px solid #EAEAEA;
        border-radius: 8px;
        padding: 16px 22px;
        margin-bottom: 16px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 12px;
    }
    .editorial-title {
        font-family: 'Newsreader', Georgia, serif;
        font-size: 24px;
        font-weight: 700;
        color: #111111;
        letter-spacing: -0.015em;
        line-height: 1.2;
    }
    .editorial-subtitle {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 13px;
        color: #444444;
    }
    .editorial-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        padding: 4px 10px;
        border-radius: 9999px;
        background: #EAEAEA;
        color: #111111;
        border: 1px solid #CCCCCC;
    }
    .editorial-dot { width: 7px; height: 7px; border-radius: 50%; background-color: #2D6A4F; }

    /* Cream Container Cards */
    .editorial-card {
        background: #FFFFFF;
        border: 1px solid #EAEAEA;
        border-radius: 8px;
        padding: 18px 22px;
        margin-bottom: 16px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        color: #111111;
        font-family: 'Newsreader', Georgia, serif;
        font-size: 15.5px;
        line-height: 1.6;
    }
    .editorial-card-title {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 12px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #555555;
        margin-bottom: 8px;
    }
    .citation-chip {
        display: inline-flex;
        align-items: center;
        background: #EAEAEA;
        color: #111111;
        border: 1px solid #CCCCCC;
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        font-weight: 600;
        padding: 2px 7px;
        border-radius: 4px;
    }

    /* Metric & Telemetry Strip */
    .editorial-metric-strip {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 10px;
        background: #FFFFFF;
        border: 1px solid #EAEAEA;
        border-radius: 10px;
        padding: 12px 16px;
        margin-bottom: 16px;
        box-shadow: 0 2px 8px rgba(58, 34, 25, 0.04);
    }
    .editorial-metric-item { display: flex; flex-direction: column; }
    .editorial-metric-label {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 11px;
        font-weight: 600;
        color: #555555;
        text-transform: uppercase;
    }
    .editorial-metric-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 16px;
        font-weight: 700;
        color: #111111;
        margin-top: 2px;
    }

    /* Modality Badges */
    .modality-badge {
        display: inline-block;
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        padding: 3px 8px;
        border-radius: 4px;
    }
    .modality-vector { background: #E4ECE3; color: #235338; border: 1px solid #BFD2BF; }
    .modality-graph { background: #E3EBF4; color: #1E3F66; border: 1px solid #BFD2E6; }
    .modality-hybrid { background: #EFE8E1; color: #533324; border: 1px solid #D6C4B8; }

    /* Streamlit Component Overrides */
    .stTextArea textarea {
        background-color: #FFFFFF !important;
        color: #111111 !important;
        border: 1px solid #CCCCCC !important;
        border-radius: 8px !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-size: 14px !important;
    }
    .stTextArea textarea:focus {
        border-color: #111111 !important;
        box-shadow: 0 0 0 2px rgba(58, 34, 25, 0.15) !important;
    }
    div.stButton > button[kind="primary"] {
        background-color: #111111 !important;
        color: #FFFFFF !important;
        border: 1px solid #111111 !important;
        border-radius: 8px !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 600 !important;
        box-shadow: 0 2px 6px rgba(58, 34, 25, 0.2) !important;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #523528 !important;
        color: #FFFFFF !important;
    }
    div.stButton > button[kind="secondary"] {
        background-color: #FFFFFF !important;
        color: #111111 !important;
        border: 1px solid #CCCCCC !important;
        border-radius: 8px !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 500 !important;
    }
    .streamlit-expanderHeader {
        background-color: #FFFFFF !important;
        color: #111111 !important;
        border: 1px solid #EAEAEA !important;
        border-radius: 8px !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 600 !important;
    }
    .streamlit-expanderContent {
        background-color: #FFFFFF !important;
        border: 1px solid #EAEAEA !important;
        border-top: none !important;
    }
    button[data-baseweb="tab"] {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        color: #444444 !important;
        font-weight: 600 !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #111111 !important;
        border-bottom-color: #111111 !important;
    }
    .editorial-timeline {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: #FFFFFF;
        border: 1px solid #EAEAEA;
        border-radius: 8px;
        padding: 12px 18px;
        margin-bottom: 16px;
    }
    .editorial-step-name { font-family: 'Plus Jakarta Sans', sans-serif; font-size: 12px; font-weight: 600; color: #111111; }
    .editorial-step-time { font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #555555; }
</style>
"""
