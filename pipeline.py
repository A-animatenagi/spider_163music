import json
import os
import shutil

from exporter import export_csvs, export_song_json
from netease import NeteaseCrawler
from sentiment import SentimentAnalyzer
from sentiment_local import LocalSentimentAnalyzer

CONFIG_FILE = "config.json"
CONFIG_EXAMPLE = "config.json.example"

DEFAULT_CONFIG = {
    "llm": {"base_url": "https://api.deepseek.com/v1", "api_key": "sk-你的APIKey",
            "model": "deepseek-chat", "timeout": 60},
    "crawler": {"request_sleep": 1.0, "max_retries": 3, "timeout": 15},
    "run": {"max_songs": 5, "comment_pages": 3, "hot_limit": 20},
    "sentiment": {"concurrency": 4, "batch_size": 12, "enabled": True,
                    "engine": "llm", "local_neg_threshold": 0.4, "local_pos_threshold": 0.6},
}


def load_config():
    created = False
    if not os.path.exists(CONFIG_FILE):
        if os.path.exists(CONFIG_EXAMPLE):
            shutil.copyfile(CONFIG_EXAMPLE, CONFIG_FILE)
        else:
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(DEFAULT_CONFIG, f, ensure_ascii=False, indent=2)
        created = True
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    for section, defaults in DEFAULT_CONFIG.items():
        cfg.setdefault(section, defaults)
        for k, v in defaults.items():
            cfg[section].setdefault(k, v)
    return cfg, created


def build_crawler(cfg):
    return NeteaseCrawler(
        sleep=cfg["crawler"]["request_sleep"],
        timeout=cfg["crawler"]["timeout"],
        max_retries=cfg["crawler"]["max_retries"],
    )


def build_sentiment(cfg):
    engine = str(cfg["sentiment"].get("engine", "llm")).strip().lower()
    if engine == "local":
        return LocalSentimentAnalyzer(
            neg_threshold=cfg["sentiment"].get("local_neg_threshold", 0.4),
            pos_threshold=cfg["sentiment"].get("local_pos_threshold", 0.6),
        )
    return SentimentAnalyzer(
        base_url=cfg["llm"]["base_url"],
        api_key=cfg["llm"]["api_key"],
        model=cfg["llm"]["model"],
        timeout=cfg["llm"]["timeout"],
        concurrency=cfg["sentiment"]["concurrency"],
        batch_size=cfg["sentiment"].get("batch_size", 12),
    )


def mark_unknown(songs, reason="情感分析未启用"):
    for song in songs:
        for c in song.get("comments") or []:
            c["sentiment"] = "unknown"
            c["sentiment_score"] = None
            c["sentiment_reason"] = reason
        SentimentAnalyzer._attach_summary(song)


def run_pipeline(cfg, keyword, limit, pages, hot_limit, out="output",
                 force_analyze=True, log=print, progress=None, song_ids=None):
    crawler = build_crawler(cfg)
    sentiment = build_sentiment(cfg)

    def report(cur, total, msg):
        if progress:
            progress(cur, total, msg)
        if log:
            log(msg)

    if song_ids:
        found = [{"song_id": sid, "name": "", "artists": [], "album": ""} for sid in song_ids]
        report(0, 4, f"[1/4] 使用已选 {len(found)} 首歌曲爬取")
    else:
        report(0, 4, f"[1/4] 搜索关键词: {keyword}")
        found = crawler.search_songs(keyword, limit=limit)
        if not found:
            raise RuntimeError("未搜索到歌曲")
        for i, s in enumerate(found, 1):
            log(f"  {i}. {s['name']} - {'/'.join(s['artists'] or [])} (专辑: {s['album']})")

    report(1, 4, f"[2/4] 抓取 {len(found)} 首歌曲的信息/专辑/歌词/评论 ...")
    songs = []
    total = len(found)
    for idx, s in enumerate(found, 1):
        full = crawler.get_song_full(s["song_id"], comment_pages=pages, hot_limit=hot_limit)
        if full:
            songs.append(full)
            log(f"  已抓取({idx}/{total}): {full['name']} | 评论 {len(full['comments'])} 条")
        else:
            log(f"  跳过(无法获取): {s['name']}")
    if not songs:
        raise RuntimeError("所有歌曲抓取失败")

    analyze = force_analyze and cfg["sentiment"]["enabled"] and sentiment.enabled
    if analyze:
        report(2, 4, f"[3/4] 情感分析（{sentiment.label}）...")
        sentiment.analyze_songs(songs)
    else:
        report(2, 4, "[3/4] 情感分析已跳过（分析未启用或依赖缺失，评论将标记为 unknown）")
        mark_unknown(songs)

    report(3, 4, f"[4/4] 导出结果到 {out} ...")
    song_dir = os.path.join(out, keyword)
    for song in songs:
        export_song_json(song, os.path.join(song_dir, f"{song['song_id']}.json"))
    songs_csv, comments_csv = export_csvs(songs, song_dir)

    report(4, 4, "完成。")
    return {
        "keyword": keyword,
        "songs": songs,
        "song_dir": song_dir,
        "songs_csv": songs_csv,
        "comments_csv": comments_csv,
        "analyzed": analyze,
    }