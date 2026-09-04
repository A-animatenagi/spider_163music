import io
import json
import os
import re

import altair as alt
import pandas as pd
import streamlit as st

from netease import NeteaseCrawler
from pipeline import build_sentiment, load_config, run_pipeline

try:
    import jieba
    from wordcloud import WordCloud
    HAS_WC = True
except Exception:
    HAS_WC = False

st.set_page_config(page_title="网易云音乐爬虫 · 评论情感分析", page_icon="🎵", layout="wide")

# ---- 情感语义色（纸面·印刷墨色：文字 ≥4.5:1 / 色块 ≥3:1）-------------
SENTIMENT_ORDER = ["正面", "中性", "负面", "未知"]
SENTIMENT_COLORS = ["#1B7A44", "#9A6B11", "#C62828", "#8C8273"]
SENTIMENT_EN = {"正面": "positive", "中性": "neutral", "负面": "negative", "未知": "unknown"}

# ---- 印刷令牌（来源见 design-system/netease-sentiment/MASTER.md）----------------
BRAND, BRAND_2, BLUE = "#E62B2B", "#C01E1E", "#2E6FB7"
POS, NEU, NEG = "#1B7A44", "#9A6B11", "#C62828"
INK_LABEL = "#1A1612"              # 图表数值/文字标签在纸面上的墨色

# ---- 内联 SVG 图标（Lucide 风格描边；规则：不用 emoji 当功能图标）---------------
_ICON = ('<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" '
         'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">{}</svg>')
ICON_MUSIC = _ICON.format('<path d="M9 18V5l12-2v13"/><circle cx="6" cy="18" r="3"/>'
                          '<circle cx="18" cy="16" r="3"/>')
ICON_COMMENT = _ICON.format('<path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2'
                            'h14a2 2 0 0 1 2 2z"/>')
ICON_GAUGE = _ICON.format('<path d="m12 14 4-4"/><path d="M3.34 19a10 10 0 1 1 17.32 0"/>')
ICON_PIE = _ICON.format('<path d="M21.21 15.89A10 10 0 1 1 8 2.83"/>'
                        '<path d="M22 12A10 10 0 0 0 12 2v10z"/>')

CSS = """
/* ====== 网易云评论情感分析 · 纸刊·乐评月刊（浅色印刷系统） ====== */
:root{
  --ny-paper:#F4EFE4; --ny-sheet:#FDFBF4; --ny-well:#ECE4D2;
  --ny-hairline:rgba(26,22,18,.18); --ny-hairline-strong:rgba(26,22,18,.45);
  --ny-text:#1A1612; --ny-text-2:#5B5346; --ny-text-3:#6E6452;
  --ny-brand:#E62B2B; --ny-brand-2:#C01E1E; --ny-blue:#2E6FB7;
  --ny-font:"PingFang SC","Microsoft YaHei","Hiragino Sans GB","Noto Sans CJK SC","Segoe UI",system-ui,sans-serif;
  --ny-mono:"Cascadia Code","JetBrains Mono","IBM Plex Mono","Consolas",ui-monospace,monospace;
}

html, body{ height:100%; background:var(--ny-paper); color:var(--ny-text); font-family:var(--ny-font); }
.stApp, [data-testid="stAppViewContainer"]{ background:var(--ny-paper); color:var(--ny-text); }
[data-testid="stHeader"]{ background:transparent; }
.block-container{ padding-top:2.1rem; max-width:1220px; }

/* ===== 侧边栏（纸面左栏 · 细竖线分隔） ===== */
[data-testid="stSidebar"]{
  background:var(--ny-paper);
  border-right:1px solid var(--ny-hairline);
}
.side-brand{ display:flex; align-items:center; gap:.6rem; padding:.15rem 0 .35rem; }
.side-mark{ width:36px;height:36px;border-radius:2px;flex:none;display:grid;place-items:center;
  background:var(--ny-brand); color:var(--ny-sheet);
  box-shadow:3px 3px 0 rgba(26,22,18,.25);}
.side-mark svg{ width:20px;height:20px;}
.side-brand b{ display:block; font-size:.98rem; color:var(--ny-text); letter-spacing:.06em; line-height:1.2; font-weight:900;}
.side-brand span{ font-size:.7rem; color:var(--ny-text-3); letter-spacing:.08em;}

/* ===== 刊头 Hero（报头：大字墨色 + 双细线 + 黑胶唱片） ===== */
.ny-hero{ display:flex; align-items:center; gap:1.5rem; padding-bottom:1rem;
  border-bottom:4px double var(--ny-text); margin:0 0 1.4rem; }
.ny-hero-txt{ min-width:0; }
.ny-over{ font-family:var(--ny-mono); font-size:.72rem; color:var(--ny-brand); letter-spacing:.2em; margin-bottom:.5rem; }
.ny-hero h1{ margin:0; font-size:2rem; font-weight:900; letter-spacing:0; line-height:1.15; color:var(--ny-text);}
.ny-hero .grad{ color:var(--ny-brand); }
.ny-hero p{ margin:.6rem 0 0; color:var(--ny-text-3); font-size:.92rem; letter-spacing:.03em;}
.ny-disc{ position:relative; width:96px;height:96px;border-radius:50%;flex:none;margin-left:auto;
  background:repeating-radial-gradient(circle at 50% 46%, #262016 0 2px, #3B3122 2px 4px, #1A1612 4px 7px);
  border:2px solid var(--ny-text); box-shadow:7px 7px 0 -3px rgba(230,43,43,.55);}
.ny-disc::after{ content:""; position:absolute; left:50%; top:50%; transform:translate(-50%,-50%);
  width:30px;height:30px;border-radius:50%;background:var(--ny-brand);
  border:3px solid var(--ny-sheet); box-shadow:0 0 0 2px var(--ny-text);}

/* ===== 标题与正文（印刷版式：墨字 + 红色栏标） ===== */
h3{ font-size:1.06rem !important; font-weight:800 !important; color:var(--ny-text) !important;
  letter-spacing:.02em; border-left:3px solid var(--ny-brand); padding-left:.6rem; }
.stCaption, .stCaption p, [data-testid="stCaptionContainer"] p, .stMarkdown small{ color:var(--ny-text-3) !important; letter-spacing:.02em;}
code, .stMarkdown code, pre, .stMarkdown pre{ font-family:var(--ny-mono); background:var(--ny-well); color:var(--ny-text);}
.stMarkdown, .stMarkdown p{ color:var(--ny-text); }
::selection{ background:var(--ny-brand); color:var(--ny-sheet); }

/* ===== KPI 数据块（印报表纸卡：细墨框 + 左侧色条 + 破折脚注） ===== */
.kpi-grid{ display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:1.1rem; margin:.4rem 0 1.8rem; }
.kpi{ --tone:var(--ny-brand); padding:.95rem 1.05rem .9rem 1.15rem; border-radius:2px;
  background:var(--ny-sheet);
  border:1px solid var(--ny-text); border-left:5px solid var(--tone);
  box-shadow:3px 3px 0 rgba(26,22,18,.07);
  animation:fadeIn .3s ease-out both;
  transition:transform .16s ease, box-shadow .16s ease;}
.kpi-ico{ color:var(--tone); display:block; }
.kpi-ico svg{ width:18px; height:18px; margin-bottom:.3rem; }
.kpi-label{ color:var(--ny-text-3); font-size:.72rem; font-weight:700; letter-spacing:.08em; }
.kpi-value{ font-size:2.1rem; font-weight:900; line-height:1.12; color:var(--ny-text);
  font-variant-numeric:tabular-nums; letter-spacing:-.02em; margin-top:.12rem;}
.kpi-sub{ color:var(--ny-text-3); font-size:.78rem; margin-top:.5rem; line-height:1.55;
  border-top:1px dashed var(--ny-hairline); padding-top:.45rem;}
.tone-brand{ --tone:var(--ny-brand) } .tone-violet{ --tone:var(--ny-blue) }
.tone-pos{ --tone:#1B7A44 } .tone-neg{ --tone:#C62828 }
.tone-neu{ --tone:#9A6B11 } .tone-unk{ --tone:#6E6757 }
.kpi:hover{ transform:translate(-2px,-2px); box-shadow:6px 6px 0 -1px rgba(230,43,43,.4); }
@keyframes fadeIn{ from{ opacity:0; transform:translateY(6px);} to{ opacity:1; transform:none;} }

/* ===== 按钮（印刷按钮：直角 + 细墨框，hover 反白；主 CTA 红块 + 硬错位影） ===== */
.stButton > button, [data-testid="stFormSubmitButton"] > button{
  border-radius:2px; font-weight:700; letter-spacing:.04em; padding:.52rem 1rem;
  border:1px solid var(--ny-text); background:var(--ny-sheet); color:var(--ny-text);
  box-shadow:none; transition:background .12s ease, color .12s ease, box-shadow .12s ease, transform .08s ease;}
.stButton > button:hover, [data-testid="stFormSubmitButton"] > button:hover{
  background:var(--ny-text); color:var(--ny-sheet); }
.stButton > button[kind="primary"], button[kind="primary"],
.stButton > button[data-testid="baseButton-primary"], button[data-testid="baseButton-primary"]{
  background:var(--ny-brand) !important; border-color:var(--ny-text) !important;
  color:var(--ny-sheet) !important; box-shadow:3px 3px 0 var(--ny-text);}
.stButton > button[kind="primary"]:hover:not(:disabled), button[kind="primary"]:hover:not(:disabled),
.stButton > button[data-testid="baseButton-primary"]:hover:not(:disabled){
  background:var(--ny-brand-2) !important; box-shadow:1px 1px 0 var(--ny-text);}
.stButton > button:active, [data-testid="stFormSubmitButton"] > button:active{ transform:translate(1px,1px); }
.stButton > button:disabled, [data-testid="stFormSubmitButton"] > button:disabled{ opacity:.45; }
.stButton > button:focus-visible, [data-testid="stFormSubmitButton"] > button:focus-visible{
  outline:2px solid var(--ny-brand); outline-offset:2px;}

/* ===== 输入控件（白纸卡片 + 墨线，聚焦转红边） ===== */
[data-testid="stTextInput"] input, [data-testid="stNumberInput"] input{
  background:var(--ny-sheet) !important; border:1px solid var(--ny-text) !important;
  border-radius:2px; color:var(--ny-text);}
[data-testid="stTextInput"] input:focus, [data-testid="stNumberInput"] input:focus{
  border-color:var(--ny-brand) !important; box-shadow:0 0 0 1px var(--ny-brand);}
[data-testid="stTextInput"] input::placeholder{ color:var(--ny-text-3); }
/* 隐藏 Streamlit 输入框/表单自带的英文回车提示（Press Enter to submit form / apply） */
[data-testid="InputInstructions"]{ display:none; }
[data-baseweb="select"] > div{ background:var(--ny-sheet) !important; border-color:var(--ny-text) !important; border-radius:2px;}
[data-baseweb="select"] > div:hover{ border-color:var(--ny-brand) !important; }

/* ===== 进度 / 状态 ===== */
[data-testid="stProgress"] > div > div > div{ background:var(--ny-brand); }
[data-testid="stStatusWidget"]{ border-radius:2px; background:var(--ny-sheet); border:1px solid var(--ny-hairline); }

/* ===== 标签页（印刷目录导航：选中 = 墨字粗体 + 红色下划线） ===== */
[data-testid="stTabs"] [data-baseweb="tab-list"]{ gap:0; border-bottom:1px solid var(--ny-text); }
button[data-baseweb="tab"]{ border-radius:0; padding:.7rem 1.05rem .72rem; font-weight:700;
  color:var(--ny-text-2); letter-spacing:.06em; margin-bottom:-1px;
  border-bottom:3px solid transparent; background:transparent !important;
  transition:color .12s ease;}
button[data-baseweb="tab"]:hover{ color:var(--ny-text); }
button[data-baseweb="tab"][aria-selected="true"]{ color:var(--ny-text); font-weight:900;
  border-bottom:3px solid var(--ny-brand); background:transparent !important;}

/* ===== 展开器 / 数据表 / 提示框（纸面细墨框，无圆角柔光） ===== */
[data-testid="stExpander"] details{ border:1px solid var(--ny-text); border-radius:2px; background:var(--ny-sheet);}
[data-testid="stExpander"] summary{ font-weight:800; }
[data-testid="stExpander"] summary svg{ fill:var(--ny-text); }
[data-testid="stDataFrame"]{ border:1px solid var(--ny-text); border-radius:2px; background:var(--ny-sheet);}
[data-testid="stDataFrame"] th{ font-size:.78rem; color:var(--ny-text-2) !important; font-weight:800 !important;
  background:var(--ny-well) !important; letter-spacing:.02em;}
[data-testid="stDataFrame"] td{ font-size:.86rem; font-variant-numeric:tabular-nums; }
[data-testid="stAlert"]{ border-radius:2px; }

/* ===== 动效降级 ===== */
@media (prefers-reduced-motion: reduce){ *{ animation:none !important; transition:none !important; } }
"""

st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)

BRAND_HEAD = ('<div class="side-brand"><span class="side-mark">{icon}</span>'
              '<div><b>网易云爬虫</b><span>评论情感分析 · 数据刊</span></div></div>').format(icon=ICON_MUSIC)

HERO_HTML = (
    '<div class="ny-hero"><div class="ny-hero-txt">'
    '<div class="ny-over">MUSIC REVIEW · 评论情感分析 · 数据刊</div>'
    '<h1>网易云音乐爬虫 <span class="grad">评论情感分析</span></h1>'
    '<p>搜索歌曲 → 抓取专辑 / 歌词 / 评论 → 情感分析（LLM · 本地双引擎）→ 可视化与导出</p>'
    '</div><div class="ny-disc" aria-hidden="true"></div></div>')


def kpi_card(label, value, sub, tone="brand", icon=ICON_MUSIC):
    return ('<div class="kpi tone-{tone}"><span class="kpi-ico">{icon}</span>'
            '<div class="kpi-label">{label}</div><div class="kpi-value">{value}</div>'
            '<div class="kpi-sub">{sub}</div></div>').format(
                tone=tone, icon=icon, label=label, value=value, sub=sub)


def kpi_row(cards):
    st.markdown('<div class="kpi-grid">' + "".join(cards) + '</div>', unsafe_allow_html=True)


def hero():
    st.markdown(HERO_HTML, unsafe_allow_html=True)

CJK_RE = re.compile(r"[\u4e00-\u9fff]")
STOPWORDS = set("的了和在是不有我你他她它这那什么就都一个也会很觉得真的喜欢歌听音乐唱专辑评论朋友人又才还对没要吧啊呀呢".strip())

LAST_STATE_FILE = os.path.join("output", ".last_session.json")


def save_last_state(res):
    if not res or not res.get("songs"):
        return
    try:
        os.makedirs(os.path.dirname(LAST_STATE_FILE), exist_ok=True)
        with open(LAST_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False)
    except Exception:
        pass


def clear_last_state():
    try:
        if os.path.exists(LAST_STATE_FILE):
            os.remove(LAST_STATE_FILE)
    except Exception:
        pass


def try_load_last_state():
    if not os.path.exists(LAST_STATE_FILE):
        return None
    try:
        with open(LAST_STATE_FILE, "r", encoding="utf-8") as f:
            res = json.load(f)
        if res and res.get("songs"):
            return res
    except Exception:
        return None
    return None


def restore_state():
    if "songs" in st.session_state and st.session_state["songs"]:
        return
    res = try_load_last_state()
    if res:
        st.session_state["songs"] = res["songs"]
        st.session_state["result"] = res
        st.session_state["keyword"] = res.get("keyword", "")
        if "_restored_notified" not in st.session_state:
            st.session_state["_restored_notified"] = True
            st.toast("已从上次运行保存恢复数据，可继续查看分析。")


def do_search(kw):
    kw = (kw or "").strip()
    if not kw:
        st.session_state["search_results"] = []
        st.session_state["search_keyword"] = ""
        st.session_state["search_error"] = "请输入搜索关键词"
        return
    crawler = NeteaseCrawler(sleep=request_sleep, timeout=cfg["crawler"]["timeout"],
                             max_retries=cfg["crawler"]["max_retries"])
    try:
        found = crawler.search_songs(kw, limit=int(max_songs))
        st.session_state["search_results"] = found
        st.session_state["search_keyword"] = kw
        st.session_state["search_error"] = None
    except Exception as e:
        st.session_state["search_results"] = []
        st.session_state["search_keyword"] = kw
        st.session_state["search_error"] = str(e)



# ---------------------------------------------------------------- helpers

def empty_song_data():
    return {"songs": [], "result": None, "keyword": ""}


def get_state(key, default=None):
    if key not in st.session_state:
        st.session_state[key] = default
    return st.session_state[key]


def cn_font_path():
    candidates = [
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\msyhbd.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\simsun.ttc",
        "/System/Library/Fonts/PingFang.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def songs_df(songs):
    rows = []
    for s in songs:
        sm = s.get("sentiment_summary") or {}
        rows.append({
            "歌曲": s.get("name"),
            "歌手": "/".join(s.get("artists") or []),
            "专辑": s.get("album"),
            "评论数": sm.get("total", 0),
            "正面": sm.get("positive", 0),
            "中性": sm.get("neutral", 0),
            "负面": sm.get("negative", 0),
            "未知": sm.get("unknown", 0),
            "平均得分": sm.get("avg_score"),
        })
    return pd.DataFrame(rows)


def count_axis(title="评论数"):
    """计数类纵轴：tickMinStep=1 强制整数刻度，避免数据量小时出现 0.2/0.4 等小数刻度。"""
    return alt.Axis(title=title, tickMinStep=1)


def display_names(names, width=14):
    """生成图表用的简短展示名：过长歌名截断加省略号，重名加 #N 后缀保证唯一。"""
    seen = {}
    out = []
    for n in names:
        base = n if len(n) <= width else n[: width - 1] + "…"
        seen[base] = seen.get(base, 0) + 1
        out.append(base if seen[base] == 1 else f"{base}#{seen[base]}")
    return out


def comments_df(songs):
    rows = []
    for s in songs:
        sname = s.get("name")
        for c in s.get("comments") or []:
            rows.append({
                "歌曲": sname,
                "昵称": c.get("nickname"),
                "评论内容": c.get("content"),
                "情感": c.get("sentiment"),
                "情感得分": c.get("sentiment_score"),
                "点赞": c.get("liked_count"),
                "时间": c.get("time_str"),
                "IP属地": c.get("ip_location"),
                "热评": "是" if c.get("is_hot") else "否",
                "时间戳": c.get("time"),
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["时间戳"] = pd.to_numeric(df["时间戳"], errors="coerce")
    return df


def word_tokens(text):
    if not text:
        return []
    return [w for w in jieba.lcut(text) if len(w) >= 2 and w not in STOPWORDS and CJK_RE.search(w)]


def _wc_color(word, *args, **kwargs):
    # wordcloud 新版以关键字参数调用 color_func（font_size/position/orientation/...），
    # 故需同时接受 *args 与 **kwargs，否则报 unexpected keyword argument 'font_size'。
    # 纸刊印刷墨色词云：红/蓝/墨/绿/褐/琥珀，在米白纸面可读
    palette = ("#C62828", "#2E6FB7", "#1A1612", "#1B7A44", "#8A5A2B", "#9A6B11", "#6E6757")
    return palette[sum(word.encode("utf-8")) % len(palette)]


def build_wordcloud_png(words, width=820, height=430):
    font = cn_font_path()
    wc = WordCloud(font_path=font, width=width, height=height,
                   background_color=None, mode="RGBA", max_words=80,
                   collocations=False, random_state=42, color_func=_wc_color)
    wc.generate(" ".join(words))
    img = wc.to_image()
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------- sidebar

cfg, _ = load_config()
restore_state()
sidebar = st.sidebar
sidebar.markdown(BRAND_HEAD, unsafe_allow_html=True)

with sidebar.form(key="search_form", clear_on_submit=False):
    keyword = st.text_input("搜索关键词", value=get_state("keyword", ""),
                            placeholder="如：毛不易 消愁")
    submitted = st.form_submit_button("搜索", type="primary", width="stretch")
sidebar.caption("输入歌名/歌手后按回车，或点击「搜索」。")

sidebar.subheader("爬取参数")
max_songs = sidebar.number_input("搜索歌曲数", 1, 30, int(cfg["run"]["max_songs"]),
                                 help="搜索并爬取的最大歌曲数量")
comment_pages = sidebar.number_input("评论页数（每页20条）", 1, 50, int(cfg["run"]["comment_pages"]),
                                     help="每首歌抓取的普通评论页数，越多越慢")
hot_limit = sidebar.number_input("热评条数", 1, 100, int(cfg["run"]["hot_limit"]),
                                 help="额外抓取的点赞最多热评条数")
request_sleep = sidebar.number_input("请求间隔（秒）", 0.0, 10.0,
                                     float(cfg["crawler"]["request_sleep"]), 0.5,
                                     help="每次请求间的等待时间，用于降低反爬风险，建议≥1")
sidebar.caption("说明：以上参数控制每次爬取的规模与速度。")

sidebar.divider()
# 情感引擎切换（方案 A：仅当前会话生效，不写回 config.json）
_ENGINE_OPTIONS = ["local", "llm"]
_ENGINE_LABELS = {"local": "本地 SnowNLP", "llm": "大模型 (LLM)"}
_current_engine = str(cfg["sentiment"].get("engine", "llm")).strip().lower()
_default_idx = _ENGINE_OPTIONS.index(_current_engine) if _current_engine in _ENGINE_OPTIONS else 1
engine_choice = sidebar.radio(
    "情感引擎",
    _ENGINE_OPTIONS,
    index=_default_idx,
    horizontal=True,
    format_func=lambda e: _ENGINE_LABELS[e],
    key="engine_choice",
    help="选择本次运行使用的情感分析引擎。切换后无需重启，立即对下一次爬取生效；仅当前会话有效，不会修改 config.json。",
)
cfg["sentiment"]["engine"] = engine_choice

sent = build_sentiment(cfg)
sidebar.caption("默认值来自 config.json 的 sentiment.engine；此处切换仅当前会话生效，不写入配置文件。")
if sent.engine == "local":
    if sent.enabled:
        sidebar.success("**本地 SnowNLP** 已启用\n\n无需 API Key，完全离线分析")
    else:
        sidebar.warning("**未安装 SnowNLP**\n\n请执行 `pip install snownlp` 后重启。")
else:
    if sent.enabled:
        sidebar.success(f"**大模型引擎** 已启用\n\n模型：{sent.model}\n\n接口：{sent.base_url}")
    else:
        sidebar.warning("**未配置有效 api_key**，评论将标记为 unknown。\n\n"
                        "请编辑项目根目录 config.json 的 llm 字段，"
                        "或将 sentiment.engine 改为 local 使用本地分析。")

if sidebar.button("清空结果", width="stretch"):
    for k in ["songs", "result", "keyword", "search_results", "search_keyword"]:
        st.session_state.pop(k, None)
    clear_last_state()
    st.rerun()

if submitted:
    do_search(keyword)


def run_section(kw, song_ids=None, limit=None):
    log_placeholder = st.empty()
    status = st.status("正在运行...", expanded=True)
    prog = st.progress(0.0, text="准备中")

    def log(msg):
        status.write(str(msg))

    def progress(cur, total, msg):
        prog.progress(min(cur / max(total, 1), 1.0), text=str(msg))

    if limit is None:
        limit = int(auto_n) if song_ids is None else int(max_songs)

    try:
        res = run_pipeline(
            cfg=cfg, keyword=kw, limit=limit, pages=int(comment_pages),
            hot_limit=int(hot_limit), out="output",
            force_analyze=True, log=log, progress=progress, song_ids=song_ids,
        )
        st.session_state["songs"] = res["songs"]
        st.session_state["result"] = res
        st.session_state["keyword"] = kw
        save_last_state(res)
        prog.progress(1.0, text="完成")
        status.update(label="运行完成", state="complete", expanded=False)
        st.success("运行完成，结果已保存（刷新浏览器后仍会保留）。请切换到其他标签页查看分析。")
        st.dataframe(songs_df(res["songs"]), width="stretch")
    except Exception as e:
        status.update(label="运行失败", state="error", expanded=True)
        st.error(f"运行失败：{e}")
    finally:
        log_placeholder.empty()


# ---------------------------------------------------------------- main tabs

hero()

t1, t2, t3, t4, t5 = st.tabs(["搜索/爬取", "情感总览", "评论分析", "词云", "导出"])


# ==================== Tab1 搜索与爬取 ====================

with t1:
    c_search, c_run = st.columns([1, 1])

    with c_search:
        st.subheader("搜索歌曲")
        st.caption("在左侧边栏输入歌名/歌手，按回车或点「搜索」；搜索结果将显示在此处。")
        search_error = st.session_state["search_error"] if "search_error" in st.session_state else None
        if search_error:
            st.error(f"搜索失败：{search_error}")
        if not keyword.strip():
            st.info("提示：请搜索网易云有版权的音乐")

        results = get_state("search_results", [])
        if results:
            st.success(f"搜索到 {len(results)} 首，请勾选需要爬取的歌曲")
            sdf = pd.DataFrame([
                {"序号": i + 1, "歌名": r["name"], "歌手": "/".join(r["artists"] or []),
                 "专辑": r["album"]}
                for i, r in enumerate(results)
            ])
            sdf.index = [r["song_id"] for r in results]
            ev = st.dataframe(sdf, width="stretch", on_select="rerun",
                              selection_mode="multi-row", key="song_select", hide_index=True)
            sel_ids = [results[int(i)]["song_id"] for i in (ev.selection.rows or [])]
            if st.button("爬取所选歌曲", type="primary", width="stretch",
                         disabled=not sel_ids):
                run_section(keyword.strip(), song_ids=sel_ids)

    with c_run:
        st.subheader("直接爬取")
        auto_n = st.number_input("自动爬取前 N 首", 1, 20, 3, key="auto_n")
        if st.button("开始爬取（前 N 首）", type="primary", width="stretch",
                     disabled=not keyword.strip()):
            run_section(keyword.strip())

    st.divider()
    st.subheader("运行日志与进度")
    _result = st.session_state["result"] if "result" in st.session_state else None
    if _result:
        st.success(f"上次运行完成：关键词「{_result['keyword']}」，共 {len(_result['songs'])} 首歌"
                   + ("，已完成情感分析。" if _result["analyzed"] else "，情感分析未启用。"))
        st.caption(f"导出目录：`{_result['song_dir']}`")


# 数据读取放在 Tab1 之后，确保本次运行中 run_section 写入的结果能立即被其它标签页使用
songs = get_state("songs", [])
result = get_state("result", None)


# ==================== Tab2 情感总览 ====================

with t2:
    if not songs:
        st.info("还没有数据，请先在「搜索/爬取」页运行一次爬取。")
    else:
        df = songs_df(songs)
        total_comments = int(df["评论数"].sum())
        avg_all = df["平均得分"].dropna().mean()
        pos_sum = int(df["正面"].sum())
        neg_sum = int(df["负面"].sum())
        n_neu = int(df["中性"].sum())
        n_unk = int(df["未知"].sum())
        pos_pct = pos_sum / max(total_comments, 1)
        neg_pct = neg_sum / max(total_comments, 1)
        score_txt = f"{avg_all:.3f}" if avg_all == avg_all else "—"
        kpi_row([
            kpi_card("歌曲数", f"{len(songs)}", "本次关键词已收录的歌曲",
                     "brand", ICON_MUSIC),
            kpi_card("评论总数", f"{total_comments:,}", "热评 + 普通评论去重后合计",
                     "violet", ICON_COMMENT),
            kpi_card("平均情感得分", score_txt,
                     "0 负面 · 0.5 中性 · 1 正面",
                     "pos" if (avg_all == avg_all and avg_all >= 0.5) else "neg",
                     ICON_GAUGE),
            kpi_card("正面 / 负面", f"{pos_pct:.0%} / {neg_pct:.0%}",
                     f"中性 {n_neu/max(total_comments,1):.0%} · 未知 {n_unk/max(total_comments,1):.0%}",
                     "neu", ICON_PIE),
        ])

        unknown_ratio = int(df["未知"].sum()) / max(total_comments, 1)
        if unknown_ratio > 0.4:
            st.warning(
                f"**注意**：当前有 **{unknown_ratio:.0%}** 的评论情感为「未知」，"
                "因此各图表可能呈现单一颜色、无法区分情感。"
                "常见原因：① `config.json` 中 `llm.api_key` 未填写或无效；② 所选模型未遵循指定 JSON 格式；"
                "③ 网络/接口异常导致分析失败。\n\n"
                "请确认已填写真实的大模型 `api_key`，并在「评论分析」页查看具体评论的情感字段。"
            )

        col_pick, _ = st.columns([1, 2])
        sel_song = col_pick.selectbox("选择歌曲", df["歌曲"].tolist(), key="dist_song")
        song_row = df[df["歌曲"] == sel_song].iloc[0]
        song_cn = " / ".join(str(df[df["歌曲"] == sel_song].iloc[0]["歌手"]).split("/"))
        st.subheader(f"情感分布 · {sel_song}（{song_cn}）")
        dist_data = pd.DataFrame({
            "情感": SENTIMENT_ORDER,
            "数量": [int(song_row[k]) for k in SENTIMENT_ORDER],
        })
        dist_data["占比"] = dist_data["数量"] / max(dist_data["数量"].sum(), 1)
        dist_data = dist_data[dist_data["数量"] > 0]
        if dist_data.empty:
            st.info("该歌曲暂无评论数据。")
        else:
            color = alt.Color("情感", scale=alt.Scale(domain=SENTIMENT_ORDER,
                                                      range=SENTIMENT_COLORS))
            col_b, col_p = st.columns([3, 2])
            with col_b:
                st.caption("评论数量")
                bars = alt.Chart(dist_data).mark_bar().encode(
                    x=alt.X("情感", sort=SENTIMENT_ORDER, axis=alt.Axis(labelAngle=0)),
                    y=alt.Y("数量", axis=count_axis()),
                    color=color,
                    tooltip=["情感", "数量", alt.Tooltip("占比", format=".1%")],
                ).properties(height=320)
                labels = alt.Chart(dist_data).mark_text(dy=-12, color=INK_LABEL).encode(
                    x=alt.X("情感", sort=SENTIMENT_ORDER),
                    y=alt.Y("数量"),
                    text="数量",
                )
                st.altair_chart(bars + labels, width="stretch")
            with col_p:
                st.caption("情感占比")
                pie = alt.Chart(dist_data).mark_arc().encode(
                    theta=alt.Theta("数量", stack=True),
                    color=alt.Color("情感", scale=alt.Scale(domain=SENTIMENT_ORDER,
                                                            range=SENTIMENT_COLORS)),
                    tooltip=["情感", "数量", alt.Tooltip("占比", format=".1%")],
                ).properties(height=320)
                st.altair_chart(pie, width="stretch")

        if len(df) > 1:
            with st.expander("各歌曲情感对比（分组柱状图）"):
                gdf = df.copy()
                gdf["简称"] = display_names(gdf["歌曲"])
                melt = gdf.melt(id_vars=["歌曲", "简称"], value_vars=SENTIMENT_ORDER,
                                var_name="情感", value_name="数量")
                melt = melt[melt["数量"] > 0]
                cmap = alt.Color("情感", scale=alt.Scale(domain=SENTIMENT_ORDER,
                                                         range=SENTIMENT_COLORS))
                gchart = alt.Chart(melt).mark_bar().encode(
                    x=alt.X("简称", sort=gdf["简称"].tolist(), title="歌曲",
                            axis=alt.Axis(labelAngle=-45, labelLimit=110, labelFontSize=10)),
                    xOffset="情感",
                    y=alt.Y("数量", axis=count_axis()),
                    color=cmap,
                    tooltip=["歌曲", "情感", "数量"],
                ).properties(height=360)
                st.altair_chart(gchart, width="stretch")
                st.caption("横轴为重名去重后的歌曲简称，悬停柱形可查看完整歌名。")

        st.subheader("各歌曲平均情感得分对比")
        avg_df = df[df["平均得分"].notna()].copy()
        if avg_df.empty:
            st.info("暂无有效的情感得分（评论情感均为「未知」或情感分析未执行）。")
        else:
            # 横向柱状图：纵轴放歌曲简称，避免歌名过长/过多时横轴标签错位、重叠
            avg_df["简称"] = display_names(avg_df["歌曲"])
            avg_df = avg_df.sort_values("平均得分", ascending=False)
            y_sort = avg_df["简称"].tolist()
            bar_h = max(len(avg_df) * 26, 240)
            avg_chart = alt.Chart(avg_df).mark_bar().encode(
                y=alt.Y("简称", sort=y_sort, title="歌曲",
                        axis=alt.Axis(labelLimit=240, labelFontSize=11)),
                x=alt.X("平均得分", scale=alt.Scale(domain=[0, 1]),
                        axis=alt.Axis(format=".2f")),
                color=alt.condition(
                    alt.datum["平均得分"] >= 0.5,
                    alt.value(POS), alt.value(NEG)),
                tooltip=["歌曲", "歌手", alt.Tooltip("平均得分", format=".3f")],
            ).properties(height=bar_h)
            avg_labels = alt.Chart(avg_df).mark_text(dx=26, align="left", color=INK_LABEL).encode(
                y=alt.Y("简称", sort=y_sort),
                x=alt.X("平均得分"),
                text=alt.Text("平均得分", format=".2f"),
            )
            st.altair_chart(avg_chart + avg_labels, width="stretch")
            st.caption("按平均得分从高到低排序；纵轴为重名去重后的歌曲简称，悬停柱形可查看完整歌名与歌手。")


# ==================== Tab3 评论分析 ====================

with t3:
    if not songs:
        st.info("还没有数据，请先在「搜索/爬取」页运行一次爬取。")
    else:
        cdf = comments_df(songs)
        song_names = cdf["歌曲"].unique().tolist()
        f_song = st.multiselect("歌曲筛选", song_names, default=song_names)
        f_sent = st.multiselect("情感筛选", ["positive", "neutral", "negative", "unknown"],
                                default=["positive", "neutral", "negative", "unknown"])
        f_text = st.text_input("评论内容关键词", "")
        f_likes = st.slider("最低点赞数", 0, int(cdf["点赞"].max() or 0), 0)

        view = cdf[cdf["歌曲"].isin(f_song) & cdf["情感"].isin(f_sent)]
        if f_text:
            view = view[view["评论内容"].str.contains(f_text, case=False, na=False)]
        view = view[view["点赞"] >= f_likes]

        sort_map = {
            "按点赞数从高到低（热度）": ("点赞", False),
            "按时间最新": ("时间戳", False),
            "按情感得分从高到低": ("情感得分", False),
        }
        sort_choice = st.selectbox("排序方式", list(sort_map.keys()), key="comment_sort")
        sort_col, sort_asc = sort_map[sort_choice]
        view = view.sort_values(sort_col, ascending=sort_asc, na_position="last")
        view.insert(0, "序号", range(1, len(view) + 1))

        st.subheader(f"评论明细（{len(view)} 条）")
        st.caption("默认按点赞数从高到低（热度优先），可切换排序方式。")
        st.dataframe(view, width="stretch", hide_index=True,
                     column_config={"序号": st.column_config.NumberColumn(width="small"),
                                    "评论内容": st.column_config.TextColumn(width="large")})

        col_l, col_r = st.columns(2)
        with col_l:
            st.subheader("情感得分分布")
            scored = view[view["情感得分"].notna()]
            if not scored.empty:
                hist = alt.Chart(scored).mark_bar().encode(
                    alt.X("情感得分", bin=alt.Bin(maxbins=20)),
                    y=alt.Y("count()", axis=count_axis()),
                    color=alt.value(POS),
                ).properties(height=280)
                st.altair_chart(hist, width="stretch")
            else:
                st.info("无情感得分数据")
        with col_r:
            st.subheader("评论时间分布")
            dated = view[view["时间戳"].notna()]
            if not dated.empty:
                dated = dated.copy()
                dated["日期"] = pd.to_datetime(dated["时间戳"], unit="ms").dt.date.astype(str)
                time_chart = alt.Chart(dated).mark_bar().encode(
                    x=alt.X("日期", title="日期"),
                    y=alt.Y("count()", axis=count_axis()),
                    color=alt.value(BLUE),
                ).properties(height=280)
                st.altair_chart(time_chart, width="stretch")
            else:
                st.info("无时间数据")

        st.subheader("热评排行（按点赞数）")
        hot = view.sort_values("点赞", ascending=False).head(20)
        st.dataframe(hot[["歌曲", "昵称", "评论内容", "点赞", "情感", "情感得分", "时间", "IP属地"]],
                     width="stretch",
                     column_config={"评论内容": st.column_config.TextColumn(width="large")})


# ==================== Tab4 词云 ====================

with t4:
    if not HAS_WC:
        st.error("缺少依赖，请执行 `pip install -r requirements.txt` 安装 jieba 与 wordcloud。")
    elif not songs:
        st.info("还没有数据，请先在「搜索/爬取」页运行一次爬取。")
    else:
        df = songs_df(songs)
        sel = st.selectbox("选择歌曲", df["歌曲"].tolist(), key="wc_song")
        sent_filter = st.radio("词云范围", ["全部评论", "仅正面", "仅负面", "仅中性", "仅未知"],
                               horizontal=True, index=0)
        song = next(s for s in songs if s["name"] == sel)
        texts = []
        for c in song.get("comments") or []:
            sent_map = {"仅正面": "positive", "仅负面": "negative",
                        "仅中性": "neutral", "仅未知": "unknown"}
            if sent_filter != "全部评论" and c.get("sentiment") != sent_map[sent_filter]:
                continue
            texts.append(c.get("content") or "")
        words = word_tokens(" ".join(texts))
        if not words:
            st.info("该范围下没有足够的分词数据。")
        else:
            with st.spinner("正在生成词云..."):
                try:
                    png = build_wordcloud_png(words)
                    st.image(png, caption=f"词云 · {sel} · {sent_filter}（{len(words)} 个词）",
                             width="stretch")
                except Exception as e:
                    st.error(f"词云生成失败：{e}")


# ==================== Tab5 导出 ====================

with t5:
    if not result:
        st.info("运行一次爬取后，这里可以下载导出的 CSV / JSON 文件。")
    else:
        d = result["song_dir"]
        st.subheader("导出目录")
        st.code(d)
        st.caption("CSV 使用 UTF-8-sig 编码，Excel 可直接打开。")

        files = []
        if os.path.exists(result["songs_csv"]):
            files.append(("songs.csv（歌曲汇总）", result["songs_csv"]))
        if os.path.exists(result["comments_csv"]):
            files.append(("comments.csv（评论明细）", result["comments_csv"]))
        if os.path.isdir(d):
            for f in sorted(os.listdir(d)):
                if f.endswith(".json"):
                    files.append((f, os.path.join(d, f)))

        for label, path in files:
            try:
                with open(path, "rb") as f:
                    st.download_button(label=f"下载 {label}", data=f.read(),
                                       file_name=os.path.basename(path), mime=None,
                                       width="stretch")
            except Exception as e:
                st.warning(f"读取 {path} 失败：{e}")

        st.subheader("情感分析汇总")
        for s in songs:
            sm = s.get("sentiment_summary") or {}
            total = max(sm.get("total", 0), 1)
            st.markdown(
                f"**{s.get('name')}** · 评论 {sm.get('total', 0)} 条 · "
                f"正面 {sm.get('positive', 0)} ({sm.get('positive', 0)/total:.0%}) · "
                f"中性 {sm.get('neutral', 0)} ({sm.get('neutral', 0)/total:.0%}) · "
                f"负面 {sm.get('negative', 0)} ({sm.get('negative', 0)/total:.0%}) · "
                f"平均得分 {sm.get('avg_score', '—')}"
            )