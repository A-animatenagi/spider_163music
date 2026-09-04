# -*- coding: utf-8 -*-
"""本地情感分析引擎的临时验证脚本"""
from sentiment_local import LocalSentimentAnalyzer
from pipeline import build_sentiment, load_config

samples = [
    ("这首歌真的太好听了，单曲循环一整天", "positive"),
    ("难听死了，简直浪费时间", "negative"),
    ("还行吧，一般般", "neutral"),
    ("唱出了我的心声，听哭了", "positive"),
    ("垃圾，什么玩意", "negative"),
    ("不好听，不推荐", "negative"),
    ("十年了，还是那么好听，青春回来了", "positive"),
    # 无词典/表情信号 → SnowNLP 低置信向中性收缩，弱通用正面句判中性（机制 4）
    ("今天天气不错", "neutral"),
    ("", "unknown"),
    # 新增机制验证：转折子句 / 多处命中 / 表情 / 水军标记 / 双重否定
    ("旋律还行，但歌词太烂了", "negative"),
    ("循环循环再循环，百听不厌", "positive"),
    ("👍👍", "positive"),
    ("🤮", "negative"),
    ("打卡路过", "neutral"),
    ("前排沙发，日报报到", "neutral"),
    ("不是不好听，是太难听了", "negative"),
]

a = LocalSentimentAnalyzer()
ok = 0
for text, expect in samples:
    r = a.analyze_one({"content": text})
    mark = "OK" if r["sentiment"] == expect else "!!"
    if r["sentiment"] == expect:
        ok += 1
    print(f"[{mark}] {r['sentiment']:8s} score={r['score']} | {text}")
    print(f"      reason: {r['reason']}")
print(f"\n准确率: {ok}/{len(samples)}")

song = {"name": "test", "comments": [{"content": t} for t, _ in samples]}
a.analyze_songs([song])
print("汇总:", song["sentiment_summary"])

cfg, _ = load_config()
cfg["sentiment"]["engine"] = "local"
print("engine=local ->", type(build_sentiment(cfg)).__name__)
cfg["sentiment"]["engine"] = "llm"
print("engine=llm   ->", type(build_sentiment(cfg)).__name__)
