"""
AURUM — Gold Price Intelligence Dashboard
BI-LSTM · BI-GRU | Return-based Deep Learning | v5.0
"""

import io, os, warnings, random
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import streamlit as st

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED); random.seed(SEED); tf.random.set_seed(SEED)

def styled_table(headers, rows, max_height=380):
    """Render styled dark HTML table. rows: list of list of (value, color) or plain value."""
    TH_BASE = "padding:10px 12px;font-family:JetBrains Mono,monospace;font-size:0.68rem;letter-spacing:0.1em;text-transform:uppercase;white-space:nowrap;color:#f0c050;background:#161b27;"
    head_cells = ""
    for i, h in enumerate(headers):
        align = "left" if i == 0 else "right"
        head_cells += "<th style='" + TH_BASE + "text-align:" + align + ";'>" + str(h) + "</th>"
    body = ""
    for row in rows:
        cells = ""
        for i, cell in enumerate(row):
            val, color = cell if isinstance(cell, tuple) else (cell, "#e8edf8")
            align = "left" if i == 0 else "right"
            TD = "padding:8px 12px;font-family:JetBrains Mono,monospace;font-size:0.78rem;white-space:nowrap;"
            cells += "<td style='" + TD + "text-align:" + align + ";color:" + color + ";'>" + str(val) + "</td>"
        body += "<tr style='border-bottom:1px solid #1d2335;'>" + cells + "</tr>"
    html = (
        "<div style='border:1px solid #1d2335;border-radius:12px;overflow:auto;max-height:" + str(max_height) + "px;margin-bottom:10px;'>"
        "<table style='width:100%;border-collapse:collapse;'>"
        "<thead style='position:sticky;top:0;z-index:2;'>"
        "<tr>" + head_cells + "</tr>"
        "</thead>"
        "<tbody style='background:#0d0f14;'>" + body + "</tbody>"
        "</table></div>"
    )
    return html

# ─── Design Tokens — AURUM palette ────────────────────────────────────────
BG        = "#07080b"
SURFACE   = "#0d0f14"
CARD      = "#12151e"
CARD2     = "#161b27"
BORDER    = "#1d2335"
BORDER2   = "#252e44"
GOLD_DEEP = "#b8832a"
GOLD_MID  = "#d4a843"
GOLD      = "#f0c050"
GOLD_LITE = "#fde99a"
GOLD_GLOW = "#f5c84222"
CREAM     = "#fdf6e3"
SILVER    = "#c0cfe0"
BLUE_ACE  = "#4a8fff"
EMERALD   = "#2ec99a"
RUBY      = "#f05060"
TEXT      = "#e8edf8"
TEXTD     = "#7a8aaa"
MUTED     = "#4a5570"

st.set_page_config(
    page_title="Gold Price Intelligence",
    page_icon="🏅",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS + Animations ─────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;600;700&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

*, html, body {{ box-sizing:border-box; margin:0; padding:0; }}

html, body, [class*="css"] {{
    background-color: {BG};
    color: {TEXT};
    font-family: 'Inter', sans-serif;
}}
.stApp {{ background-color:{BG}; }}

/* ── Particle canvas background ── */
#particle-canvas {{
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    pointer-events: none;
    z-index: 0;
    opacity: 0.55;
}}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, {SURFACE} 0%, {BG} 100%);
    border-right: 1px solid {BORDER};
    z-index: 10;
}}
[data-testid="stSidebar"] label  {{ color:{TEXTD} !important; font-size:0.8rem !important; }}
[data-testid="stSidebarContent"] {{ padding-top: 1.5rem; }}

/* Slider thumb gold */
[data-testid="stSlider"] div[role="slider"] {{
    background:{GOLD} !important; border-color:{GOLD} !important;
}}
[data-testid="stSlider"] div[data-testid="stSliderTrack"] > div:first-child {{
    background: linear-gradient(90deg, {GOLD_DEEP},{GOLD}) !important;
}}

/* ── Metrics ── */
[data-testid="stMetric"] {{
    background: linear-gradient(145deg, {CARD2} 0%, {CARD} 100%);
    border: 1px solid {BORDER};
    border-top: 2px solid transparent;
    border-image: linear-gradient(90deg,{GOLD_DEEP},{GOLD},{GOLD_DEEP}) 1 0 0 0;
    border-radius: 14px;
    padding: 18px 22px 16px;
    position: relative;
    overflow: hidden;
    transition: transform 0.2s, box-shadow 0.2s;
}}
[data-testid="stMetric"]:hover {{
    transform: translateY(-2px);
    box-shadow: 0 8px 32px {GOLD_GLOW};
}}
[data-testid="stMetric"]::after {{
    content:'';
    position:absolute; top:0; right:0;
    width:60px; height:60px;
    background: radial-gradient(circle at top right, {GOLD}18, transparent 70%);
}}
[data-testid="stMetricLabel"] {{
    color:{MUTED} !important;
    font-family:'JetBrains Mono',monospace !important;
    font-size:0.68rem !important;
    letter-spacing:0.12em;
    text-transform:uppercase;
}}
[data-testid="stMetricValue"] {{
    color:{GOLD_LITE} !important;
    font-family:'JetBrains Mono',monospace !important;
    font-size:1.4rem !important;
    font-weight:600 !important;
}}
[data-testid="stMetricDelta"] {{ font-size:0.8rem !important; font-weight:500 !important; }}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {{
    background:{SURFACE};
    border:1px solid {BORDER};
    border-radius:12px;
    padding:5px 6px;
    gap:4px;
}}
.stTabs [data-baseweb="tab"] {{
    border-radius:8px;
    color:{MUTED};
    font-size:0.83rem;
    font-weight:600;
    padding:8px 20px;
    letter-spacing:0.03em;
    border:none;
    transition: all 0.2s;
}}
.stTabs [aria-selected="true"] {{
    background: linear-gradient(135deg, {GOLD}1a, {GOLD_DEEP}22) !important;
    border:1px solid {GOLD}40 !important;
    color:{GOLD} !important;
    box-shadow: 0 0 12px {GOLD}18;
}}

/* ── Buttons ── */
.stButton>button {{
    background: linear-gradient(135deg, {GOLD_DEEP} 0%, {GOLD} 50%, {GOLD_DEEP} 100%);
    background-size: 200% 100%;
    color:{BG};
    border:none;
    border-radius:10px;
    padding:12px 24px;
    font-weight:700;
    font-family:'Inter',sans-serif;
    font-size:0.9rem;
    letter-spacing:0.04em;
    width:100%;
    transition: background-position 0.4s, transform 0.2s, box-shadow 0.2s;
    cursor:pointer;
}}
.stButton>button:hover {{
    background-position: 100% 0;
    transform:translateY(-2px);
    box-shadow:0 8px 28px {GOLD}55;
}}

/* ── Progress bar ── */
[data-testid="stProgress"] > div > div {{
    background: linear-gradient(90deg,{GOLD_DEEP},{GOLD},{GOLD_LITE}) !important;
}}

/* ── Expander ── */
[data-testid="stExpander"] {{
    background:{CARD};
    border:1px solid {BORDER} !important;
    border-radius:12px !important;
}}
details summary {{ color:{TEXTD} !important; font-size:0.87rem !important; }}

/* ── Select / number input ── */
[data-baseweb="select"]>div {{
    background:{SURFACE} !important;
    border-color:{BORDER2} !important;
    border-radius:8px !important;
    color:{TEXT} !important;
}}
[data-baseweb="input"] input {{
    background:{SURFACE} !important;
    color:{TEXT} !important;
    border-color:{BORDER2} !important;
    border-radius:8px !important;
    font-family:'JetBrains Mono',monospace !important;
}}
[data-baseweb="textarea"] textarea {{
    background:{SURFACE} !important;
    color:{TEXT} !important;
}}

/* ── Download button ── */
[data-testid="stDownloadButton"]>button {{
    background:{SURFACE} !important;
    color:{GOLD} !important;
    border:1px solid {GOLD}44 !important;
    border-radius:10px !important;
    font-weight:600 !important;
    transition: all 0.2s !important;
}}
[data-testid="stDownloadButton"]>button:hover {{
    background:{GOLD}12 !important;
    border-color:{GOLD}99 !important;
    box-shadow:0 4px 16px {GOLD}33 !important;
}}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {{
    border:1px solid {BORDER};
    border-radius:12px;
    overflow:hidden;
}}

hr {{ border-color:{BORDER}; margin:12px 0; }}

/* ════════════════════════════════
   CUSTOM COMPONENTS
════════════════════════════════ */

/* Hero banner */
.aurum-hero {{
    position:relative;
    background: linear-gradient(135deg, {CARD2} 0%, {CARD} 40%, {SURFACE} 100%);
    border:1px solid {BORDER};
    border-radius:20px;
    padding:36px 40px 32px;
    margin-bottom:24px;
    overflow:hidden;
}}
.aurum-hero::before {{
    content:'';
    position:absolute;
    top:-80px; right:-80px;
    width:320px; height:320px;
    background: radial-gradient(circle, {GOLD}14 0%, transparent 65%);
    pointer-events:none;
    animation: pulse-glow 4s ease-in-out infinite;
}}
.aurum-hero::after {{
    content:'Au';
    position:absolute;
    bottom:-30px; right:40px;
    font-family:'Cinzel',serif;
    font-size:10rem;
    font-weight:700;
    color:{GOLD}08;
    line-height:1;
    pointer-events:none;
    user-select:none;
}}
@keyframes pulse-glow {{
    0%,100% {{ opacity:0.7; transform:scale(1); }}
    50%      {{ opacity:1;   transform:scale(1.08); }}
}}

.hero-eyebrow {{
    font-family:'JetBrains Mono',monospace;
    font-size:0.68rem;
    letter-spacing:0.2em;
    color:{GOLD};
    text-transform:uppercase;
    margin-bottom:12px;
    display:flex;
    align-items:center;
    gap:8px;
}}
.hero-eyebrow::before {{
    content:'';
    display:inline-block;
    width:24px; height:1px;
    background:{GOLD};
}}
.hero-title {{
    font-family:'Cinzel',serif;
    font-size:2.6rem;
    font-weight:700;
    line-height:1.1;
    margin-bottom:10px;
    background: linear-gradient(135deg, {GOLD_LITE} 0%, {GOLD} 40%, {GOLD_DEEP} 100%);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
    background-clip:text;
}}
.hero-sub {{
    color:{TEXTD};
    font-size:0.92rem;
    line-height:1.7;
    max-width:580px;
}}
.hero-tags {{
    display:flex;
    gap:8px;
    flex-wrap:wrap;
    margin-top:18px;
}}
.hero-tag {{
    font-family:'JetBrains Mono',monospace;
    font-size:0.7rem;
    letter-spacing:0.08em;
    color:{GOLD};
    background:{GOLD}12;
    border:1px solid {GOLD}30;
    border-radius:6px;
    padding:4px 10px;
    text-transform:uppercase;
}}

/* Live price ticker */
.price-ticker {{
    background: linear-gradient(90deg, {CARD2}, {CARD});
    border:1px solid {BORDER};
    border-radius:14px;
    padding:16px 24px;
    display:flex;
    align-items:center;
    gap:32px;
    flex-wrap:wrap;
    margin-bottom:22px;
    position:relative;
    overflow:hidden;
}}
.price-ticker::before {{
    content:'';
    position:absolute;
    left:0; top:0; bottom:0;
    width:3px;
    background: linear-gradient(180deg,{GOLD_DEEP},{GOLD},{GOLD_DEEP});
}}
.ticker-item {{ display:flex; flex-direction:column; gap:3px; }}
.ticker-label {{
    font-family:'JetBrains Mono',monospace;
    font-size:0.62rem;
    letter-spacing:0.14em;
    color:{MUTED};
    text-transform:uppercase;
}}
.ticker-value {{
    font-family:'JetBrains Mono',monospace;
    font-size:1.1rem;
    font-weight:600;
    color:{GOLD_LITE};
}}
.ticker-value.up   {{ color:{EMERALD}; }}
.ticker-value.down {{ color:{RUBY}; }}
.ticker-sep {{ width:1px; background:{BORDER}; align-self:stretch; }}

/* Section headers */
.sec-eyebrow {{
    font-family:'JetBrains Mono',monospace;
    font-size:0.62rem;
    letter-spacing:0.16em;
    color:{GOLD};
    text-transform:uppercase;
    margin-bottom:5px;
}}
.sec-title {{
    font-size:1.05rem;
    font-weight:700;
    color:{TEXT};
    border-bottom:1px solid {BORDER};
    padding-bottom:10px;
    margin-bottom:14px;
}}

/* Gold shimmer badge */
.badge-aurum {{
    background: linear-gradient(135deg, {GOLD_DEEP}, {GOLD}, {GOLD_DEEP});
    background-size:200% 100%;
    color:{BG};
    font-weight:700;
    padding:5px 18px;
    border-radius:20px;
    font-size:0.8rem;
    display:inline-block;
    font-family:'JetBrains Mono',monospace;
    letter-spacing:0.06em;
    animation: shimmer 2.5s ease-in-out infinite;
}}
@keyframes shimmer {{
    0%,100% {{ background-position:0% 50%; }}
    50%      {{ background-position:100% 50%; }}
}}

.badge-model-lstm {{
    background:{RUBY}18; color:{RUBY};
    border:1px solid {RUBY}40;
    font-weight:600; padding:4px 12px;
    border-radius:20px; font-size:0.75rem;
    display:inline-block;
    font-family:'JetBrains Mono',monospace;
}}
.badge-model-gru {{
    background:{EMERALD}18; color:{EMERALD};
    border:1px solid {EMERALD}40;
    font-weight:600; padding:4px 12px;
    border-radius:20px; font-size:0.75rem;
    display:inline-block;
    font-family:'JetBrains Mono',monospace;
}}

.info-banner {{
    background:{CARD};
    border:1px solid {BORDER};
    border-left:3px solid {BLUE_ACE};
    border-radius:12px;
    padding:18px 22px;
    color:{TEXTD};
    font-size:0.88rem;
    line-height:1.7;
}}

/* Sidebar card */
.sb-card {{
    background:{CARD};
    border:1px solid {BORDER};
    border-radius:12px;
    padding:14px 16px;
    margin-bottom:14px;
}}
.sb-card-title {{
    font-family:'JetBrains Mono',monospace;
    font-size:0.65rem;
    letter-spacing:0.14em;
    text-transform:uppercase;
    color:{GOLD};
    font-weight:600;
    margin-bottom:12px;
    display:flex; align-items:center; gap:6px;
}}

/* Gold divider */
.gold-divider {{
    height:1px;
    background: linear-gradient(90deg, transparent, {GOLD}55, transparent);
    margin:20px 0;
    border:none;
}}

/* Floating gold coin animation */
@keyframes float-coin {{
    0%,100% {{ transform: translateY(0px) rotate(0deg); }}
    33%      {{ transform: translateY(-8px) rotate(3deg); }}
    66%      {{ transform: translateY(-4px) rotate(-2deg); }}
}}
.float-coin {{
    display:inline-block;
    animation: float-coin 3s ease-in-out infinite;
}}

/* Stat pill */
.stat-pill {{
    display:inline-flex; align-items:center; gap:6px;
    background:{CARD}; border:1px solid {BORDER};
    border-radius:8px; padding:6px 14px;
    font-family:'JetBrains Mono',monospace;
    font-size:0.78rem; color:{TEXTD};
    margin:3px 4px 3px 0;
}}
.stat-pill b {{ color:{TEXT}; }}
.stat-pill .dot {{
    width:6px; height:6px; border-radius:50%;
    background:{GOLD}; flex-shrink:0;
}}

/* Glow ring around section */
.glow-section {{
    background: linear-gradient({CARD}, {CARD}) padding-box,
                linear-gradient(135deg, {GOLD_DEEP}44, transparent, {GOLD}22) border-box;
    border:1px solid transparent;
    border-radius:16px;
    padding:22px 24px;
    margin-bottom:18px;
}}

/* Table style */
.stDataFrame th {{
    background:{CARD2} !important;
    color:{GOLD} !important;
    font-family:'JetBrains Mono',monospace !important;
    font-size:0.75rem !important;
}}

/* Scrollbar */
::-webkit-scrollbar {{ width:6px; height:6px; }}
::-webkit-scrollbar-track {{ background:{SURFACE}; }}
::-webkit-scrollbar-thumb {{ background:{GOLD_DEEP}; border-radius:3px; }}
::-webkit-scrollbar-thumb:hover {{ background:{GOLD}; }}
</style>
""", unsafe_allow_html=True)

# ─── Particle canvas injected via components (script must run outside st.markdown) ──
import streamlit.components.v1 as components
components.html("""
<canvas id="pc" style="position:fixed;top:0;left:0;width:100vw;height:100vh;pointer-events:none;z-index:0;opacity:0.5;"></canvas>
<script>
(function() {
    var c = document.getElementById('pc');
    if (!c) return;
    var ctx = c.getContext('2d');
    var W = window.innerWidth, H = window.innerHeight;
    c.width = W; c.height = H;
    window.addEventListener('resize', function() { W=window.innerWidth; H=window.innerHeight; c.width=W; c.height=H; });
    var GC = ['#f0c050','#d4a843','#fde99a','#b8832a','#e8c878'];
    var ps = [];
    for (var i=0; i<55; i++) {
        var sz = Math.random()*2.4+0.6;
        ps.push({ x:Math.random()*W, y:Math.random()*H, vx:(Math.random()-0.5)*0.35, vy:(Math.random()-0.5)*0.22-0.08,
            sz:sz, bsz:sz, a:Math.random()*0.5+0.15, col:GC[Math.floor(Math.random()*GC.length)],
            p:Math.random()*Math.PI*2, ps:Math.random()*0.025+0.01, t:Math.random()>0.7?'d':'c' });
    }
    function dd(x,y,s) { ctx.beginPath(); ctx.moveTo(x,y-s); ctx.lineTo(x+s,y); ctx.lineTo(x,y+s); ctx.lineTo(x-s,y); ctx.closePath(); }
    function animate() {
        ctx.clearRect(0,0,W,H);
        for (var i=0; i<ps.length; i++) {
            var p = ps[i];
            p.x+=p.vx; p.y+=p.vy; p.p+=p.ps;
            var pf=1+Math.sin(p.p)*0.3, ds=p.bsz*pf, da=p.a*(0.8+Math.sin(p.p)*0.2);
            if(p.x<-10)p.x=W+10; if(p.x>W+10)p.x=-10; if(p.y<-10)p.y=H+10; if(p.y>H+10)p.y=-10;
            ctx.save(); ctx.globalAlpha=da; ctx.fillStyle=p.col; ctx.shadowBlur=ds*4; ctx.shadowColor=p.col;
            if(p.t==='d') { dd(p.x,p.y,ds); } else { ctx.beginPath(); ctx.arc(p.x,p.y,ds,0,Math.PI*2); }
            ctx.fill(); ctx.restore();
        }
        requestAnimationFrame(animate);
    }
    animate();
})();
</script>
""", height=0, scrolling=False)

# ─── Matplotlib theme — premium dark ──────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor"  : CARD,
    "axes.facecolor"    : CARD,
    "axes.edgecolor"    : BORDER2,
    "axes.labelcolor"   : TEXTD,
    "xtick.color"       : TEXTD,
    "ytick.color"       : TEXTD,
    "text.color"        : TEXT,
    "grid.color"        : BORDER,
    "grid.linewidth"    : 0.5,
    "figure.dpi"        : 120,
    "axes.spines.top"   : False,
    "axes.spines.right" : False,
    "axes.titlesize"    : 11,
    "axes.titleweight"  : "bold",
    "axes.titlecolor"   : TEXT,
    "font.family"       : "sans-serif",
})

# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"""
    <div style="text-align:center;padding:10px 0 20px;">
        <div style="display:flex;justify-content:center;margin-bottom:8px;">
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 140 85" width="130" height="79"
             style="filter:drop-shadow(0 5px 20px #f0c05088);">
          <defs>
            <linearGradient id="sbTop" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%"   stop-color="#fef5c0"/>
              <stop offset="40%"  stop-color="#f5c842"/>
              <stop offset="100%" stop-color="#9a6008"/>
            </linearGradient>
            <linearGradient id="sbFront" x1="0%" y1="0%" x2="0%" y2="100%">
              <stop offset="0%"   stop-color="#e8a820"/>
              <stop offset="55%"  stop-color="#c07c10"/>
              <stop offset="100%" stop-color="#6a4008"/>
            </linearGradient>
            <linearGradient id="sbRight" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%"   stop-color="#c89020"/>
              <stop offset="100%" stop-color="#7a5010"/>
            </linearGradient>
            <linearGradient id="sbShine" x1="0%" y1="0%" x2="80%" y2="100%">
              <stop offset="0%"   stop-color="#ffffff" stop-opacity="0.55"/>
              <stop offset="100%" stop-color="#ffffff" stop-opacity="0"/>
            </linearGradient>
          </defs>
          <style>
            .sb-gb {{ animation: sb-fl 3.2s ease-in-out infinite; transform-origin:70px 52px; }}
            @keyframes sb-fl {{
              0%,100% {{ transform:translateY(0px) rotate(-0.8deg); }}
              30%      {{ transform:translateY(-6px) rotate(0.8deg); }}
              70%      {{ transform:translateY(-3px) rotate(-0.4deg); }}
            }}
          </style>
          <g class="sb-gb">
            <ellipse cx="72" cy="78" rx="48" ry="5" fill="#000" opacity="0.22"/>
            <polygon points="105,47 124,36 124,60 105,71" fill="url(#sbRight)"/>
            <polygon points="15,47 105,47 105,71 15,71" fill="url(#sbFront)"/>
            <polygon points="15,47 105,47 124,36 34,36" fill="url(#sbTop)"/>
            <polygon points="34,36 82,36 92,42 44,44" fill="url(#sbShine)" opacity="0.65"/>
            <line x1="15"  y1="47" x2="105" y2="47" stroke="#fde99a" stroke-width="0.7" opacity="0.5"/>
            <line x1="34"  y1="36" x2="124" y2="36" stroke="#fde99a" stroke-width="0.5" opacity="0.3"/>
            <line x1="15"  y1="47" x2="34"  y2="36" stroke="#fde99a" stroke-width="0.5" opacity="0.3"/>
            <line x1="105" y1="47" x2="124" y2="36" stroke="#fde99a" stroke-width="0.5" opacity="0.3"/>
            <polygon points="42,39 98,39 106,43 50,44" fill="none" stroke="#c8980a" stroke-width="0.8" opacity="0.5"/>
            <text x="72" y="62" text-anchor="middle" font-family="Georgia,serif"
                  font-size="8.5" font-weight="bold" fill="#fde99a" letter-spacing="2.5" opacity="0.88">GOLD BAR</text>
            <text x="72" y="68.5" text-anchor="middle" font-family="monospace"
                  font-size="5.5" fill="#f0c050" letter-spacing="1" opacity="0.65">AU 999.9 · 24K</text>
            <line x1="15" y1="71" x2="105" y2="71" stroke="#5a3808" stroke-width="0.8" opacity="0.5"/>
          </g>
        </svg>
        </div>
        <div style="font-family:JetBrains Mono,monospace;font-size:0.6rem;
            letter-spacing:0.18em;color:{MUTED};text-transform:uppercase;
            margin-top:2px;">Gold Price Intelligence</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f'<div class="sb-card"><div class="sb-card-title">📂 Dataset</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="font-family:JetBrains Mono,monospace;font-size:0.72rem;color:{TEXTD};line-height:2;">
        📄 data_lengkap_ML_aligned_v2.xlsx<br>
        <span style="color:{MUTED};">Sheet: Data Aligned · Auto-loaded</span><br>
        <span style="color:{EMERALD};">✅ Data siap digunakan</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(f'<div class="sb-card"><div class="sb-card-title">🧠 Parameter Model</div>', unsafe_allow_html=True)
    window_size   = st.slider("Window Size (lookback)", 10, 60, 30)
    forecast_days = st.slider("Hari Prediksi", 7, 180, 30)
    epochs        = st.slider("Max Epoch", 30, 200, 100, 10)
    batch_size    = st.selectbox("Batch Size", [8,16,32], index=1)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(f'<div class="sb-card"><div class="sb-card-title">🔬 Pilih Model</div>', unsafe_allow_html=True)
    run_lstm = st.checkbox("BI-LSTM", value=True)
    run_gru  = st.checkbox("BI-GRU",  value=True)
    st.markdown('</div>', unsafe_allow_html=True)

    train_btn = st.button("⚡ Mulai Training", type="primary")

    st.markdown(f"""
    <div style="margin-top:16px;padding:12px 14px;
        background:linear-gradient({CARD},{SURFACE});
        border:1px solid {BORDER};border-radius:10px;text-align:center;">
        <div style="font-family:JetBrains Mono,monospace;font-size:0.62rem;
            color:{MUTED};line-height:1.8;">
            Gold Predictor v5.0 · Return-based DL<br>
            BI-LSTM · BI-GRU · MinMaxScaler<br>
            <span style="color:{GOLD};">Gold Price Forecasting</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# LOAD DATA — auto dari folder data/
# ═══════════════════════════════════════════════════════════════════════════
FEATURES_RAW = ["Gold","Batu Bara","Nikel","USD Index","Tembaga","Inflasi Global","Bitcoin","Perak"]

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "data_lengkap_ML_aligned_v2.xlsx")
try:
    df_raw = pd.read_excel(DATA_PATH, sheet_name="Data Aligned")
except Exception as e:
    st.error(f"❌ Gagal membaca data: {e}"); st.stop()

df_raw = df_raw.rename(columns={
    "tanggal":"Tanggal","usd index":"USD Index","USD index":"USD Index",
    "inflasi global":"Inflasi Global","Inflasi global":"Inflasi Global","perak":"Perak"
})
if "Tanggal" not in df_raw.columns: st.error("Kolom 'Tanggal' tidak ditemukan."); st.stop()
if "Gold"    not in df_raw.columns: st.error("Kolom 'Gold' tidak ditemukan."); st.stop()

if pd.api.types.is_numeric_dtype(df_raw["Tanggal"]):
    df_raw["Tanggal"] = pd.to_datetime(df_raw["Tanggal"], unit="D", origin="1899-12-30")
else:
    df_raw["Tanggal"] = pd.to_datetime(df_raw["Tanggal"], dayfirst=True, errors="coerce")
# Drop rows where tanggal failed to parse
df_raw = df_raw.dropna(subset=["Tanggal"])
# Ensure proper datetime64[ns] — avoids matplotlib OverflowError
df_raw["Tanggal"] = pd.to_datetime(df_raw["Tanggal"].dt.strftime("%Y-%m-%d"))

FEAT_AVAIL = [f for f in FEATURES_RAW if f in df_raw.columns]
df_model   = df_raw[["Tanggal"]+FEAT_AVAIL].copy().sort_values("Tanggal").reset_index(drop=True)
for col in FEAT_AVAIL:
    df_model[col] = pd.to_numeric(df_model[col], errors="coerce")
df_model[FEAT_AVAIL] = df_model[FEAT_AVAIL].ffill().bfill()
for col in FEAT_AVAIL:
    df_model[col+"_Return"] = df_model[col].pct_change()
FEAT_MODEL = [f+"_Return" for f in FEAT_AVAIL]
df_model   = df_model.dropna().reset_index(drop=True)

# ── Hero with data ─────────────────────────────────────────────────────────
gold_last = df_model["Gold"].iloc[-1]
gold_prev = df_model["Gold"].iloc[-2]
gold_chg  = gold_last - gold_prev
gold_pct  = gold_chg / gold_prev * 100
chg_cls   = "up" if gold_chg >= 0 else "down"
chg_arrow = "▲" if gold_chg >= 0 else "▼"

st.markdown(f"""
<div class="aurum-hero">
    <div class="hero-eyebrow">Deep Learning · Time Series · Commodity Intelligence</div>
    <div class="hero-title">Gold Price<br>Intelligence</div>
    <div class="hero-sub">
        Model <b style="color:{TEXT}">BI-LSTM</b> & <b style="color:{TEXT}">BI-GRU</b>
        berbasis <b style="color:{GOLD}">Return Target</b> —
        {len(df_model):,} observasi ·
        {df_model['Tanggal'].iloc[0].strftime('%b %Y')} –
        {df_model['Tanggal'].iloc[-1].strftime('%b %Y')}
    </div>
    <div class="hero-tags">
        <span class="hero-tag">Bidirectional LSTM</span>
        <span class="hero-tag">Bidirectional GRU</span>
        <span class="hero-tag">Return-Based</span>
        <span class="hero-tag">{len(FEAT_AVAIL)} Variabel</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Price ticker ───────────────────────────────────────────────────────────
vol_30 = df_model["Gold_Return"].tail(30).std()*100
yoy    = ((gold_last / df_model["Gold"].iloc[max(0,len(df_model)-260)]) - 1)*100 if len(df_model)>260 else 0

st.markdown(f"""
<div class="price-ticker">
    <div class="ticker-item">
        <div class="ticker-label">🥇 Harga Terakhir</div>
        <div class="ticker-value">${gold_last:,.2f}</div>
    </div>
    <div class="ticker-sep"></div>
    <div class="ticker-item">
        <div class="ticker-label">Perubahan Harian</div>
        <div class="ticker-value {chg_cls}">{chg_arrow} ${abs(gold_chg):,.2f} ({gold_pct:+.2f}%)</div>
    </div>
    <div class="ticker-sep"></div>
    <div class="ticker-item">
        <div class="ticker-label">Tertinggi Sepanjang Masa</div>
        <div class="ticker-value">${df_model['Gold'].max():,.2f}</div>
    </div>
    <div class="ticker-sep"></div>
    <div class="ticker-item">
        <div class="ticker-label">Terendah</div>
        <div class="ticker-value">${df_model['Gold'].min():,.2f}</div>
    </div>
    <div class="ticker-sep"></div>
    <div class="ticker-item">
        <div class="ticker-label">Volatilitas 30H</div>
        <div class="ticker-value">{vol_30:.2f}%</div>
    </div>
    <div class="ticker-sep"></div>
    <div class="ticker-item">
        <div class="ticker-label">Return YoY</div>
        <div class="ticker-value {'up' if yoy>=0 else 'down'}">{yoy:+.1f}%</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════
tab1,tab2,tab3,tab4,tab5 = st.tabs([
    "📊  Eksplorasi Data",
    "🔴  BI-LSTM",
    "🟢  BI-GRU",
    "⚖️  Perbandingan",
    "🔮  Prediksi",
])

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1
# ═══════════════════════════════════════════════════════════════════════════
with tab1:
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("📈 All-Time High",   f"${df_model['Gold'].max():,.0f}")
    c2.metric("📉 All-Time Low",    f"${df_model['Gold'].min():,.0f}")
    c3.metric("📊 Rata-rata",        f"${df_model['Gold'].mean():,.0f}")
    c4.metric("📐 Std Deviasi",      f"${df_model['Gold'].std():,.0f}")
    c5.metric("📅 Total Observasi",  f"{len(df_model):,}")

    st.markdown("<br>", unsafe_allow_html=True)

    col_ctrl,_ = st.columns([1,3])
    with col_ctrl:
        rentang_opt = st.selectbox("Rentang tampilan",
            ["Semua Data","6 Bulan Terakhir","1 Tahun Terakhir","2 Tahun Terakhir"])
    nm = {"Semua Data":len(df_model),"6 Bulan Terakhir":130,"1 Tahun Terakhir":260,"2 Tahun Terakhir":520}
    n_show = min(nm[rentang_opt], len(df_model))
    dfs = df_model.tail(n_show)

    # Main price + return chart
    fig, axes = plt.subplots(2,1,figsize=(13,6.5),
                              gridspec_kw={"height_ratios":[2.8,1],"hspace":0.06})
    # Price panel
    ax = axes[0]
    gold_vals = dfs["Gold"].values
    dates_arr = dfs["Tanggal"].values
    ax.plot(dates_arr, gold_vals, color=GOLD, linewidth=2, zorder=4)
    # Gradient fill via poly
    ax.fill_between(dates_arr, gold_vals, gold_vals.min()*0.995,
                    alpha=0.10, color=GOLD, zorder=2)
    # 20-day MA
    if len(dfs)>20:
        ma20 = dfs["Gold"].rolling(20).mean()
        ax.plot(dates_arr, ma20.values, color=GOLD_DEEP, linewidth=1.2,
                linestyle="--", alpha=0.7, label="MA-20")
        ax.legend(fontsize=8, facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT)
    ax.set_ylabel("Harga Emas (USD)", labelpad=10)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"${x:,.0f}"))
    ax.tick_params(labelbottom=False); ax.grid(True, alpha=0.2)
    ax.set_title("Pergerakan Harga Emas & Moving Average", pad=10)
    # Annotate high/low
    imax_local = np.argmax(gold_vals); imin_local = np.argmin(gold_vals)
    ax.annotate(f"${gold_vals[imax_local]:,.0f}",
                xy=(dates_arr[imax_local], gold_vals[imax_local]),
                fontsize=8, color=GOLD, fontweight="bold",
                xytext=(0,8), textcoords="offset points", ha="center")
    # Return panel
    ax2 = axes[1]
    rvals = dfs["Gold_Return"].values
    bar_colors = [EMERALD if v>=0 else RUBY for v in rvals]
    ax2.bar(dates_arr, rvals, color=bar_colors, alpha=0.75, width=1.2, zorder=3)
    ax2.axhline(0, color=BORDER2, linewidth=0.8)
    ax2.set_ylabel("Daily Return", labelpad=10)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x*100:.1f}%"))
    ax2.grid(True, alpha=0.18)
    fig.autofmt_xdate(rotation=25)
    plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)

    col_l,col_r = st.columns([1.2,1])
    with col_l:
        st.markdown('<div class="sec-eyebrow">Statistik</div>', unsafe_allow_html=True)
        st.markdown('<div class="sec-title">Ringkasan Deskriptif</div>', unsafe_allow_html=True)
        COMMODITY_ICONS2 = {"Gold":"🥇","Batu Bara":"⚫","Nikel":"🔩","USD Index":"💵",
                            "Tembaga":"🟠","Inflasi Global":"📈","Bitcoin":"₿","Perak":"🥈","Crude Oil":"🛢️"}
        desc = df_model[FEAT_AVAIL].describe().T.round(2).rename(
            columns={"count":"N","mean":"Mean","std":"Std","min":"Min",
                     "25%":"Q1","50%":"Median","75%":"Q3","max":"Max"})
        stat_rows = []
        for var in desc.index:
            r = desc.loc[var]
            icon = COMMODITY_ICONS2.get(var, "📊")
            stat_rows.append([
                (icon + " " + var, "#fde99a"),
                (str(int(r["N"])), "#7a8aaa"),
                ("%.2f" % r["Mean"], "#e8edf8"),
                ("%.2f" % r["Std"], "#c0cfe0"),
                ("%.2f" % r["Min"], "#f05060"),
                ("%.2f" % r["Q1"], "#7a8aaa"),
                ("%.2f" % r["Median"], "#f0c050"),
                ("%.2f" % r["Q3"], "#7a8aaa"),
                ("%.2f" % r["Max"], "#2ec99a"),
            ])
        st.markdown(styled_table(
            ["Variabel","N","Mean","Std","Min","Q1","Median","Q3","Max"],
            stat_rows, max_height=320
        ), unsafe_allow_html=True)

    with col_r:
        st.markdown('<div class="sec-eyebrow">Korelasi</div>', unsafe_allow_html=True)
        st.markdown('<div class="sec-title">Hubungan terhadap Gold</div>', unsafe_allow_html=True)
        corr = df_model[FEAT_AVAIL].corr()["Gold"].drop("Gold").sort_values(ascending=False)
        fig3, ax3 = plt.subplots(figsize=(5.5,4))
        cb = [EMERALD if v>=0 else RUBY for v in corr.values]
        bars3 = ax3.barh(corr.index[::-1], corr.values[::-1],
                         color=cb[::-1], alpha=0.82, height=0.55, zorder=3)
        ax3.axvline(0, color=BORDER2, linewidth=1)
        ax3.set_xlabel("Pearson Correlation"); ax3.grid(True,axis="x",alpha=0.18)
        for bar, val in zip(bars3, corr.values[::-1]):
            off = 0.012 if val>=0 else -0.012
            ax3.text(val+off, bar.get_y()+bar.get_height()/2,
                     f"{val:.3f}", va="center",
                     ha="left" if val>=0 else "right",
                     fontsize=8.5, color=TEXT, fontweight="600")
        ax3.set_title("Korelasi Pearson terhadap Gold"); ax3.grid(True,axis="x",alpha=0.18)
        plt.tight_layout(); st.pyplot(fig3); plt.close(fig3)

        # Correlation strength table — styled
        COMMODITY_ICONS = {"Batu Bara":"⚫","Nikel":"🔩","USD Index":"💵","Tembaga":"🟠",
                           "Inflasi Global":"📈","Bitcoin":"₿","Perak":"🥈","Crude Oil":"🛢️"}
        corr_rows = []
        for var, rv in zip(corr.index, corr.values):
            rv_r = round(rv, 3)
            bar_len = int(abs(rv_r) * 12)
            bar_fill = "█" * bar_len + "░" * (12 - bar_len)
            bar_color = "#2ec99a" if rv_r >= 0 else "#f05060"
            strength = "🟢 Kuat" if abs(rv_r)>=0.6 else ("🟡 Sedang" if abs(rv_r)>=0.3 else "🔴 Lemah")
            arah = "↑ Positif" if rv_r >= 0 else "↓ Negatif"
            icon = COMMODITY_ICONS.get(var, "📊")
            corr_rows.append([
                (icon + " " + var, "#fde99a"),
                (("%+.3f" % rv_r), bar_color),
                (bar_fill, bar_color),
                (strength, "#e8edf8"),
                (arah, bar_color),
            ])
        st.markdown(styled_table(
            ["Variabel", "r", "Visual", "Kekuatan", "Arah"],
            corr_rows, max_height=280
        ), unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════
def create_sequences(X_data, y_data, ws):
    X,y,idx=[],[],[]
    for i in range(ws,len(X_data)):
        X.append(X_data[i-ws:i,:]); y.append(y_data[i]); idx.append(i)
    return np.array(X),np.array(y),np.array(idx)

def eval_metrics(y_a,y_p):
    mae  = mean_absolute_error(y_a,y_p)
    rmse = np.sqrt(mean_squared_error(y_a,y_p))
    mape = np.mean(np.abs((y_a-y_p)/y_a))*100
    r2   = r2_score(y_a,y_p)
    return {"MAE":mae,"RMSE":rmse,"MAPE (%)":mape,"Akurasi (%)":100-mape,"R²":r2}

if "trained" not in st.session_state:
    st.session_state["trained"] = False

if train_btn:
    if not run_lstm and not run_gru:
        st.warning("Pilih minimal satu model."); st.stop()

    scaler   = MinMaxScaler()
    X_scaled = scaler.fit_transform(df_model[FEAT_MODEL])
    y_return = df_model["Gold_Return"].values
    X,y,idx_tgt = create_sequences(X_scaled, y_return, window_size)
    split = int(len(X)*0.8)
    X_train,X_test = X[:split],X[split:]
    y_train,y_test = y[:split],y[split:]
    idx_test = idx_tgt[split:]
    date_test        = df_model["Tanggal"].iloc[idx_test].reset_index(drop=True)
    gold_actual_test = df_model["Gold"].iloc[idx_test].reset_index(drop=True)
    gold_prev_test   = df_model["Gold"].iloc[idx_test-1].reset_index(drop=True)
    y_test_actual    = gold_actual_test.values
    n_feat           = X_train.shape[2]
    results          = {}
    es = EarlyStopping(monitor="val_loss",patience=15,restore_best_weights=True)

    if run_lstm:
        pg = st.progress(0, text="⚡ Melatih BI-LSTM...")
        tf.random.set_seed(SEED); np.random.seed(SEED)
        m=Sequential([
            Bidirectional(LSTM(64,return_sequences=True),input_shape=(window_size,n_feat)),
            Dropout(0.2),
            Bidirectional(LSTM(32,return_sequences=False)),
            Dropout(0.2),Dense(16,activation="relu"),Dense(1)
        ],name="BI_LSTM")
        m.compile(optimizer=Adam(0.001),loss="mse")
        h=m.fit(X_train,y_train,epochs=epochs,batch_size=batch_size,
                validation_split=0.2,callbacks=[es],shuffle=False,verbose=0)
        yr=m.predict(X_test,verbose=0).reshape(-1)
        yp=gold_prev_test.values*(1+yr)
        results["BI-LSTM"]={"model":m,"history":h.history,"pred_price":yp,"metrics":eval_metrics(y_test_actual,yp)}
        pg.progress(100,text="✅ BI-LSTM selesai!")

    if run_gru:
        pg2=st.progress(0,text="⚡ Melatih BI-GRU...")
        tf.random.set_seed(SEED); np.random.seed(SEED)
        m2=Sequential([
            Bidirectional(GRU(64,return_sequences=True),input_shape=(window_size,n_feat)),
            Dropout(0.2),
            Bidirectional(GRU(32,return_sequences=False)),
            Dropout(0.2),Dense(16,activation="relu"),Dense(1)
        ],name="BI_GRU")
        m2.compile(optimizer=Adam(0.001),loss="mse")
        h2=m2.fit(X_train,y_train,epochs=epochs,batch_size=batch_size,
                  validation_split=0.2,callbacks=[es],shuffle=False,verbose=0)
        yr2=m2.predict(X_test,verbose=0).reshape(-1)
        yp2=gold_prev_test.values*(1+yr2)
        results["BI-GRU"]={"model":m2,"history":h2.history,"pred_price":yp2,"metrics":eval_metrics(y_test_actual,yp2)}
        pg2.progress(100,text="✅ BI-GRU selesai!")

    best_name=min(results,key=lambda k:results[k]["metrics"]["MAPE (%)"])
    st.session_state.update({
        "trained":True,"results":results,"y_test_actual":y_test_actual,
        "date_test":date_test,"gold_prev_test":gold_prev_test,
        "X_scaled":X_scaled,"scaler":scaler,"window_size":window_size,
        "forecast_days":forecast_days,"n_feat":n_feat,"best_name":best_name,
        "df_model":df_model,"FEAT_MODEL":FEAT_MODEL,
    })

# ═══════════════════════════════════════════════════════════════════════════
# HELPER — render model tab
# ═══════════════════════════════════════════════════════════════════════════
def render_model_tab(key, color_model, badge_html):
    res    = st.session_state["results"][key]
    y_act  = st.session_state["y_test_actual"]
    date_t = st.session_state["date_test"]
    m      = res["metrics"]
    best   = st.session_state["best_name"]

    if key==best:
        st.markdown(f'<div style="margin-bottom:14px"><span class="badge-aurum">🏆 Model Terbaik</span></div>', unsafe_allow_html=True)

    st.markdown(f'<div style="margin-bottom:16px">{badge_html}</div>', unsafe_allow_html=True)

    c1,c2,c3,c4,c5=st.columns(5)
    c1.metric("R²",f"{m['R²']:.4f}")
    c2.metric("MAE",f"${m['MAE']:,.1f}")
    c3.metric("RMSE",f"${m['RMSE']:,.1f}")
    c4.metric("MAPE",f"{m['MAPE (%)']:.2f}%")
    c5.metric("Akurasi",f"{m['Akurasi (%)']:.2f}%")

    st.markdown("<br>", unsafe_allow_html=True)

    col_a,col_b=st.columns([1.2,2.8])
    with col_a:
        rt=st.selectbox("Rentang data uji",["Semua","50 Terakhir","100 Terakhir"],key=f"rt_{key}")
    n_t=min({"Semua":len(y_act),"50 Terakhir":50,"100 Terakhir":100}[rt],len(y_act))
    y_a=y_act[-n_t:]; y_p=res["pred_price"][-n_t:]; d_t=date_t.iloc[-n_t:]
    err=np.abs(y_a-y_p)

    fig,axes=plt.subplots(1,2,figsize=(14,4.5))
    # Loss
    ax=axes[0]
    epochs_done=len(res["history"]["loss"])
    xep=range(epochs_done)
    ax.plot(xep,res["history"]["loss"],     color=color_model,linewidth=2,label="Train",zorder=3)
    ax.plot(xep,res["history"]["val_loss"], color=GOLD,linewidth=1.5,linestyle="--",label="Validation",zorder=3,alpha=0.85)
    best_ep=np.argmin(res["history"]["val_loss"])
    ax.axvline(best_ep,color=GOLD,alpha=0.35,linewidth=1.2,linestyle=":")
    ax.annotate(f"best\nep.{best_ep}",
                xy=(best_ep,res["history"]["val_loss"][best_ep]),
                fontsize=7,color=GOLD,alpha=0.9,xytext=(6,10),textcoords="offset points")
    ax.set_title(f"Loss Curve — {key}"); ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss")
    ax.legend(fontsize=8,facecolor=CARD,edgecolor=BORDER,labelcolor=TEXT)
    ax.grid(True,alpha=0.2)
    # Prediction panel
    ax2=axes[1]
    ax2.fill_between(range(n_t),y_a-err,y_a+err,alpha=0.1,color=color_model,zorder=1,label="Error band")
    ax2.plot(range(n_t),y_a,color=GOLD,linewidth=2.2,label="Aktual",zorder=4)
    ax2.plot(range(n_t),y_p,color=color_model,linewidth=1.7,linestyle="--",label=f"Prediksi {key}",zorder=5)
    ax2.set_title(f"Prediksi vs Aktual — {n_t} data uji")
    ax2.set_xlabel("Index"); ax2.set_ylabel("Harga Emas (USD)")
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_:f"${x:,.0f}"))
    ax2.legend(fontsize=8,facecolor=CARD,edgecolor=BORDER,labelcolor=TEXT)
    ax2.grid(True,alpha=0.2)
    plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    # Error dist + Scatter
    col_x,col_y=st.columns(2)
    with col_x:
        st.markdown('<div class="sec-eyebrow">Distribusi Error</div><div class="sec-title">Absolute Error Histogram</div>', unsafe_allow_html=True)
        fig_e,ax_e=plt.subplots(figsize=(6,3))
        ax_e.hist(err,bins=28,color=color_model,alpha=0.75,edgecolor=BORDER,zorder=3)
        ax_e.axvline(err.mean(),color=GOLD,linewidth=1.8,linestyle="--",label=f"Mean ${err.mean():,.0f}")
        ax_e.axvline(np.median(err),color=SILVER,linewidth=1.2,linestyle=":",label=f"Median ${np.median(err):,.0f}")
        ax_e.set_xlabel("Absolute Error (USD)"); ax_e.set_ylabel("Frekuensi")
        ax_e.legend(fontsize=8,facecolor=CARD,edgecolor=BORDER,labelcolor=TEXT)
        ax_e.grid(True,alpha=0.18)
        plt.tight_layout(); st.pyplot(fig_e); plt.close(fig_e)

    with col_y:
        st.markdown('<div class="sec-eyebrow">Scatter Plot</div><div class="sec-title">Aktual vs Prediksi</div>', unsafe_allow_html=True)
        fig_s,ax_s=plt.subplots(figsize=(6,3))
        sc=ax_s.scatter(y_a,y_p,c=err,cmap="RdYlGn_r",alpha=0.55,s=14,zorder=3,vmin=0,vmax=err.max())
        mn=min(y_a.min(),y_p.min()); mx=max(y_a.max(),y_p.max())
        ax_s.plot([mn,mx],[mn,mx],color=GOLD,linewidth=1.3,linestyle="--",label="Ideal",zorder=4)
        ax_s.set_xlabel("Aktual (USD)"); ax_s.set_ylabel("Prediksi (USD)")
        ax_s.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_:f"${x:,.0f}"))
        ax_s.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_:f"${x:,.0f}"))
        ax_s.legend(fontsize=8,facecolor=CARD,edgecolor=BORDER,labelcolor=TEXT)
        ax_s.grid(True,alpha=0.18)
        plt.colorbar(sc,ax=ax_s,label="Abs Error",pad=0.02)
        plt.tight_layout(); st.pyplot(fig_s); plt.close(fig_s)

    with st.expander("📋 Tabel Prediksi Detail"):
        df_t=pd.DataFrame({
            "Tanggal":d_t.values,
            "Aktual (USD)":y_a.round(2),
            "Prediksi":y_p.round(2),
            "Error Absolut":err.round(2),
            "Error (%)": (err/y_a*100).round(3),
        })
        ep_vals = (err/y_a*100)
        detail_rows = []
        for i in range(len(df_t)):
            ep = ep_vals[i]
            ep_color = "#2ec99a" if ep < 1 else ("#f0c050" if ep < 3 else "#f05060")
            acc_icon = "🟢" if ep < 1 else ("🟡" if ep < 3 else "🔴")
            detail_rows.append([
                (str(df_t["Tanggal"].iloc[i])[:10], "#7a8aaa"),
                ("$%.2f" % y_a[i], "#fde99a"),
                ("$%.2f" % y_p[i], color_model),
                ("$%.2f" % err[i], ep_color),
                ("%.3f%%" % ep, ep_color),
                (acc_icon, ep_color),
            ])
        st.markdown(styled_table(
            ["📅 Tanggal","🥇 Aktual","🎯 Prediksi","Err Abs","Err %","Akurasi"],
            detail_rows, max_height=380
        ), unsafe_allow_html=True)
        st.download_button("⬇️ Download CSV — " + key,
                           data=df_t.to_csv(index=False).encode(),
                           file_name="prediksi_" + key.lower().replace("-","_") + ".csv",
                           mime="text/csv")

# ── TAB 2 & 3 ─────────────────────────────────────────────────────────────
with tab2:
    if not st.session_state["trained"] or "BI-LSTM" not in st.session_state.get("results",{}):
        st.markdown('<div class="info-banner">▶ Klik <b>⚡ Mulai Training</b> di sidebar (centang BI-LSTM).</div>', unsafe_allow_html=True)
    else:
        render_model_tab("BI-LSTM", RUBY,
            f'<span class="badge-model-lstm">● Bidirectional LSTM</span>')

with tab3:
    if not st.session_state["trained"] or "BI-GRU" not in st.session_state.get("results",{}):
        st.markdown('<div class="info-banner">▶ Klik <b>⚡ Mulai Training</b> di sidebar (centang BI-GRU).</div>', unsafe_allow_html=True)
    else:
        render_model_tab("BI-GRU", EMERALD,
            f'<span class="badge-model-gru">● Bidirectional GRU</span>')

# ═══════════════════════════════════════════════════════════════════════════
# TAB 4 — PERBANDINGAN
# ═══════════════════════════════════════════════════════════════════════════
with tab4:
    results=st.session_state.get("results",{})
    if not st.session_state["trained"] or len(results)<2:
        st.markdown('<div class="info-banner">▶ Latih kedua model untuk perbandingan.</div>', unsafe_allow_html=True)
    else:
        y_act=st.session_state["y_test_actual"]
        date_t=st.session_state["date_test"]
        lm=results["BI-LSTM"]["metrics"]; gm=results["BI-GRU"]["metrics"]
        best=st.session_state["best_name"]

        st.markdown(f'<div style="margin-bottom:20px">Model Terbaik: <span class="badge-aurum">🏆 {best}</span></div>', unsafe_allow_html=True)

        # Tabel perbandingan metrik — styled
        METRIC_ICONS = {"MAE":"📏","RMSE":"📐","MAPE (%)":"🎯","Akurasi (%)":"✅","R²":"📊"}
        higher_better = ["Akurasi (%)","R²"]
        cmp_rows = []
        for mk in ["MAE","RMSE","MAPE (%)","Akurasi (%)","R²"]:
            lv = lm[mk]; gv = gm[mk]
            hi = mk in higher_better
            lstm_win = (lv <= gv and not hi) or (lv >= gv and hi)
            lc = "#f05060" if lstm_win else "#7a8aaa"
            gc = "#2ec99a" if not lstm_win else "#7a8aaa"
            delta = abs(lv - gv)
            winner = ("🔴 BI-LSTM" if lstm_win else "🟢 BI-GRU")
            winner_c = "#f05060" if lstm_win else "#2ec99a"
            icon = METRIC_ICONS.get(mk, "📊")
            cmp_rows.append([
                (icon + " " + mk, "#fde99a"),
                ("%.4f%s" % (lv, " 🏆" if lstm_win else ""), lc),
                ("%.4f%s" % (gv, " 🏆" if not lstm_win else ""), gc),
                ("%.4f" % delta, "#4a5570"),
                (winner, winner_c),
            ])
        st.markdown(styled_table(
            ["Metrik","🔴 BI-LSTM","🟢 BI-GRU","Δ Selisih","Unggul"],
            cmp_rows, max_height=300
        ), unsafe_allow_html=True)

        st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)

        col_l,col_r=st.columns(2)
        with col_l:
            st.markdown('<div class="sec-title">MAPE & Akurasi</div>', unsafe_allow_html=True)
            fig,axes=plt.subplots(1,2,figsize=(7,3.5))
            for ax,metric,hi_good,c in zip(
                axes,["MAPE (%)","Akurasi (%)"],
                [False,True],
                [[RUBY,EMERALD],[RUBY,EMERALD]]
            ):
                vals=[lm[metric],gm[metric]]
                bi=np.argmin(vals) if not hi_good else np.argmax(vals)
                bc=[c[0] if i==bi else f"{c[0]}55" for i in range(2)] if metric=="MAPE (%)" \
                   else [c[1] if i==bi else f"{c[1]}55" for i in range(2)]
                bars=ax.bar(["BI-LSTM","BI-GRU"],vals,color=bc,width=0.42,zorder=3)
                for bar in bars:
                    h=bar.get_height()
                    ax.text(bar.get_x()+bar.get_width()/2,h+0.02,
                            f"{h:.2f}",ha="center",fontsize=10,color=TEXT,fontweight="700")
                ax.set_title(metric,fontsize=9); ax.grid(True,axis="y",alpha=0.2)
                mn=min(vals)
                ax.set_ylim([mn*0.97 if hi_good else 0,max(vals)*1.1])
            plt.tight_layout(); st.pyplot(fig); plt.close(fig)

        with col_r:
            st.markdown('<div class="sec-title">Radar — Semua Metrik</div>', unsafe_allow_html=True)
            metrics_r=["MAE","RMSE","MAPE (%)","Akurasi (%)","R²"]
            norm_l=np.zeros(5); norm_g=np.zeros(5)
            for i,k in enumerate(metrics_r):
                mn_v=min(lm[k],gm[k]); mx_v=max(lm[k],gm[k])
                rng=mx_v-mn_v if mx_v!=mn_v else 1
                norm_l[i]=(lm[k]-mn_v)/rng; norm_g[i]=(gm[k]-mn_v)/rng
                if k in ["MAE","RMSE","MAPE (%)"]:
                    norm_l[i]=1-norm_l[i]; norm_g[i]=1-norm_g[i]
            N=len(metrics_r)
            angles=[n/float(N)*2*3.14159 for n in range(N)]; angles+=[angles[0]]
            nl=np.append(norm_l,norm_l[0]); ng=np.append(norm_g,norm_g[0])
            fig_r,ax_r=plt.subplots(figsize=(4.8,3.5),subplot_kw=dict(polar=True))
            ax_r.set_facecolor(CARD); fig_r.patch.set_facecolor(CARD)
            ax_r.plot(angles,nl,color=RUBY,  linewidth=2,label="BI-LSTM",zorder=3)
            ax_r.fill(angles,nl,color=RUBY,  alpha=0.18,zorder=2)
            ax_r.plot(angles,ng,color=EMERALD,linewidth=2,label="BI-GRU",zorder=3)
            ax_r.fill(angles,ng,color=EMERALD,alpha=0.18,zorder=2)
            ax_r.set_xticks(angles[:-1])
            ax_r.set_xticklabels(metrics_r,size=8,color=TEXT)
            ax_r.set_yticklabels([]); ax_r.grid(color=BORDER2,linewidth=0.6)
            ax_r.spines["polar"].set_color(BORDER)
            ax_r.legend(loc="lower right",fontsize=8,
                        facecolor=CARD,edgecolor=BORDER,labelcolor=TEXT,
                        bbox_to_anchor=(1.35,0))
            plt.tight_layout(); st.pyplot(fig_r); plt.close(fig_r)

        # Overlay
        st.markdown('<div class="sec-title">Overlay Prediksi Kedua Model vs Aktual</div>', unsafe_allow_html=True)
        col_ov,_=st.columns([1,3])
        with col_ov:
            rc=st.selectbox("Rentang",["Semua","50 Terakhir","100 Terakhir"],key="r_cmp")
        n_c=min({"Semua":len(y_act),"50 Terakhir":50,"100 Terakhir":100}[rc],len(y_act))
        fig4,ax4=plt.subplots(figsize=(13,4.5))
        ax4.plot(date_t.iloc[-n_c:],y_act[-n_c:],
                 color=GOLD,linewidth=2.5,label="Aktual",zorder=5)
        ax4.plot(date_t.iloc[-n_c:],results["BI-LSTM"]["pred_price"][-n_c:],
                 color=RUBY,linewidth=1.6,linestyle="--",label="BI-LSTM",zorder=4)
        ax4.plot(date_t.iloc[-n_c:],results["BI-GRU"]["pred_price"][-n_c:],
                 color=EMERALD,linewidth=1.6,linestyle=":",label="BI-GRU",zorder=3)
        ax4.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_:f"${x:,.0f}"))
        ax4.set_ylabel("Harga Emas (USD)"); ax4.legend(fontsize=9,facecolor=CARD,edgecolor=BORDER,labelcolor=TEXT)
        ax4.grid(True,alpha=0.2); fig4.autofmt_xdate(rotation=25)
        plt.tight_layout(); st.pyplot(fig4); plt.close(fig4)

        with st.expander("📋 Tabel Perbandingan Prediksi"):
            lstm_p = results["BI-LSTM"]["pred_price"].round(2)
            gru_p  = results["BI-GRU"]["pred_price"].round(2)
            overlay_rows = []
            for i in range(len(y_act)):
                act = y_act[i]; lp = lstm_p[i]; gp = gru_p[i]
                le = abs(act - lp); ge = abs(act - gp)
                lstm_win = le <= ge
                overlay_rows.append([
                    (str(date_t.values[i])[:10], "#7a8aaa"),
                    ("$%.2f" % act, "#fde99a"),
                    ("$%.2f" % lp, "#f05060" if lstm_win else "#7a8aaa"),
                    ("$%.2f" % gp, "#2ec99a" if not lstm_win else "#7a8aaa"),
                    ("$%.2f" % le, "#f05060" if le > ge else "#2ec99a"),
                    ("$%.2f" % ge, "#f05060" if ge > le else "#2ec99a"),
                    ("🔴 LSTM" if lstm_win else "🟢 GRU", "#f05060" if lstm_win else "#2ec99a"),
                ])
            st.markdown(styled_table(
                ["📅 Tanggal","🥇 Aktual","🔴 BI-LSTM","🟢 BI-GRU","Err LSTM","Err GRU","Lebih Dekat"],
                overlay_rows, max_height=360
            ), unsafe_allow_html=True)
            df_h=pd.DataFrame({
                "Tanggal":date_t.values,
                "Gold Aktual":y_act.round(2),
                "Prediksi BI-LSTM":lstm_p,
                "Prediksi BI-GRU":gru_p,
            })
            st.download_button("⬇️ Download Hasil Testing",
                               data=df_h.to_csv(index=False).encode(),
                               file_name="hasil_testing_gabungan.csv",mime="text/csv")

# ═══════════════════════════════════════════════════════════════════════════
# TAB 5 — PREDIKSI
# ═══════════════════════════════════════════════════════════════════════════
with tab5:
    results=st.session_state.get("results",{})
    if not st.session_state["trained"] or len(results)==0:
        st.markdown('<div class="info-banner">▶ Latih model terlebih dahulu.</div>', unsafe_allow_html=True)
    else:
        best_name=st.session_state["best_name"]
        X_scaled =st.session_state["X_scaled"]
        ws       =st.session_state["window_size"]
        n_feat   =st.session_state["n_feat"]
        df_m     =st.session_state["df_model"]

        col_m,col_fd,col_ctx=st.columns([1,1,1])
        with col_m:
            model_choice=st.selectbox("Model prediksi",list(results.keys()),
                                      index=list(results.keys()).index(best_name))
        with col_fd:
            custom_fd=st.number_input("Jumlah hari prediksi",1,365,
                                      value=st.session_state["forecast_days"])
        with col_ctx:
            rh=st.selectbox("Historis ditampilkan",
                            ["30 Hari","60 Hari","90 Hari","Semua"],key="hist_ctx")

        mc=RUBY if model_choice=="BI-LSTM" else EMERALD
        sel_model=results[model_choice]["model"]

        # Forecast
        lw=X_scaled[-ws:].copy()
        last_price=df_m["Gold"].iloc[-1]
        fp=[]; cp=last_price
        for _ in range(custom_fd):
            inp=lw.reshape(1,ws,n_feat)
            pr=sel_model.predict(inp,verbose=0)[0,0]
            nxt=cp*(1+pr); fp.append(nxt); cp=nxt
            lw=np.vstack([lw[1:],lw[-1]])
        fp=np.array(fp)

        last_date=df_m["Tanggal"].iloc[-1]
        try:
            future_dates=pd.bdate_range(start=last_date+pd.Timedelta(days=1),periods=custom_fd)
            date_labels=[d.strftime("%d %b %Y") for d in future_dates]
        except:
            date_labels=[f"H+{i+1}" for i in range(custom_fd)]

        # KPI
        badge_html=f'<span class="badge-model-lstm">● BI-LSTM</span>' if model_choice=="BI-LSTM" \
                   else f'<span class="badge-model-gru">● BI-GRU</span>'
        st.markdown(f'<div class="sec-title">🔮 Prediksi {custom_fd} Hari ke Depan &nbsp; {badge_html}</div>', unsafe_allow_html=True)

        c1,c2,c3,c4=st.columns(4)
        c1.metric("Harga Aktual Terakhir",f"${last_price:,.2f}")
        c2.metric(f"H+1  ({date_labels[0]})",f"${fp[0]:,.2f}",delta=f"{fp[0]-last_price:+,.2f}")
        c3.metric(f"H+{custom_fd//2}",f"${fp[custom_fd//2-1]:,.2f}",delta=f"{fp[custom_fd//2-1]-last_price:+,.2f}")
        c4.metric(f"H+{custom_fd}  ({date_labels[-1]})",f"${fp[-1]:,.2f}",delta=f"{fp[-1]-last_price:+,.2f}")

        st.markdown("<br>", unsafe_allow_html=True)

        # Main forecast chart
        n_ctx=min({"30 Hari":30,"60 Hari":60,"90 Hari":90,"Semua":len(df_m)}[rh],len(df_m))
        hd=df_m["Tanggal"].iloc[-n_ctx:]
        hp=df_m["Gold"].iloc[-n_ctx:].values

        fig5,ax5=plt.subplots(figsize=(13,5))
        ax5.plot(hd,hp,color=GOLD,linewidth=2.2,label="Historis Aktual",zorder=4)
        from matplotlib.patches import Polygon as MPoly
        # skip polygon fill for forecast chart to avoid date overflow

        try: fx=list(future_dates)
        except: fx=list(range(custom_fd))

        fut_plot=np.concatenate([[hp[-1]],fp])
        fut_x_all=[pd.Timestamp(hd.iloc[-1])]+list(fx)
        band_lo=np.concatenate([[hp[-1]]],)*1 ; band_hi=band_lo.copy()
        band_lo=np.concatenate([[hp[-1]],fp*0.95])
        band_hi=np.concatenate([[hp[-1]],fp*1.05])
        ax5.fill_between(fut_x_all,band_lo,band_hi,alpha=0.12,color=mc,zorder=2,label="Band ±5%")
        ax5.plot(fut_x_all,fut_plot,color=mc,linewidth=2.2,
                 linestyle="--",marker="o",markersize=3.5,
                 label=f"Prediksi {model_choice}",zorder=5)
        ax5.axvline(x=hd.iloc[-1],color=BORDER2,linestyle=":",linewidth=1.5,alpha=0.8)
        ax5.set_ylabel("Harga Emas (USD)")
        ax5.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_:f"${x:,.0f}"))
        ax5.legend(fontsize=9,facecolor=CARD,edgecolor=BORDER,labelcolor=TEXT)
        ax5.grid(True,alpha=0.2); fig5.autofmt_xdate(rotation=25)
        plt.tight_layout(); st.pyplot(fig5); plt.close(fig5)

        # Daily delta bar chart
        st.markdown('<div class="sec-title">Tren Perubahan Harian</div>', unsafe_allow_html=True)
        fig6,ax6=plt.subplots(figsize=(13,2.8))
        deltas=np.concatenate([[fp[0]-last_price],np.diff(fp)])
        bcolors=[EMERALD if d>=0 else RUBY for d in deltas]
        ax6.bar(range(custom_fd),deltas,color=bcolors,alpha=0.82,width=0.75,zorder=3)
        ax6.axhline(0,color=BORDER2,linewidth=0.8)
        ax6.set_ylabel("Δ Harian (USD)")
        ax6.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_:f"{x:+,.0f}"))
        ax6.grid(True,axis="y",alpha=0.18)
        step_lbl=max(1,custom_fd//8)
        ax6.set_xticks(range(0,custom_fd,step_lbl))
        ax6.set_xticklabels([date_labels[i] for i in range(0,custom_fd,step_lbl)],
                            rotation=20,ha="right",fontsize=7.5)
        plt.tight_layout(); st.pyplot(fig6); plt.close(fig6)

        # Table + download
        df_future=pd.DataFrame({
            "Hari":range(1,custom_fd+1),
            "Tanggal":date_labels,
            "Prediksi (USD)":fp.round(2),
            "Δ Harian":deltas.round(2),
            "Δ Kumulatif":(fp-last_price).round(2),
        })
        col_tbl,col_dl=st.columns([3,1])
        with col_tbl:
            with st.expander("📋 Tabel Prediksi Lengkap"):
                cum = fp - last_price
                forecast_rows = []
                for i in range(len(fp)):
                    d_val = deltas[i]; c_val = cum[i]
                    d_color = "#2ec99a" if d_val >= 0 else "#f05060"
                    c_color = "#2ec99a" if c_val >= 0 else "#f05060"
                    trend = "📈" if d_val >= 0 else "📉"
                    forecast_rows.append([
                        (str(i+1), "#4a5570"),
                        (date_labels[i], "#7a8aaa"),
                        ("$%.2f" % fp[i], mc),
                        (("%+.2f" % d_val), d_color),
                        (("%+.2f" % c_val), c_color),
                        (trend, d_color),
                    ])
                st.markdown(styled_table(
                    ["#","📅 Tanggal","💰 Prediksi","Δ Harian","Δ Kumulatif","Tren"],
                    forecast_rows, max_height=400
                ), unsafe_allow_html=True)
        with col_dl:
            st.download_button(
                "⬇️ Download CSV",
                data=df_future.to_csv(index=False).encode(),
                file_name="prediksi_" + model_choice.lower().replace("-","_") + "_" + str(custom_fd) + "hari.csv",
                mime="text/csv",
                use_container_width=True,
            )
