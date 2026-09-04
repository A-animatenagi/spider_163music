import argparse
import os
import sys

from pipeline import CONFIG_FILE, run_pipeline, load_config


def setup_console():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def load_config_cli():
    cfg, created = load_config()
    if created:
        print(f"[提示] 已生成 {CONFIG_FILE}，请在其中填入大模型 API 配置后再运行。")
        sys.exit(0)
    return cfg


def print_song_summary(song):
    s = song.get("sentiment_summary") or {}
    print("-" * 60)
    print(f"歌曲: {song.get('name')} | 歌手: {'/'.join(song.get('artists') or [])}")
    print(f"专辑: {song.get('album')} | 评论数: {s.get('total', 0)}")
    if s.get("total"):
        pos = s.get("positive", 0)
        neu = s.get("neutral", 0)
        neg = s.get("negative", 0)
        unk = s.get("unknown", 0)
        avg = s.get("avg_score")
        total = max(s.get("total"), 1)
        print(f"情感分布: 正面 {pos} ({pos/total:.1%}) | 中性 {neu} ({neu/total:.1%})"
              f" | 负面 {neg} ({neg/total:.1%}) | 未知 {unk}")
        if avg is not None:
            print(f"平均情感得分: {avg} (0=非常负面, 0.5=中性, 1=非常正面)")


def main():
    setup_console()
    parser = argparse.ArgumentParser(description="网易云音乐爬虫 + 评论情感分析")
    parser.add_argument("keyword", nargs="?", help="搜索关键词（歌手名/歌名）")
    parser.add_argument("-n", "--limit", type=int, help="搜索返回歌曲数（默认取配置文件 max_songs）")
    parser.add_argument("-p", "--pages", type=int, help="每首歌普通评论抓取页数（每页20条）")
    parser.add_argument("-o", "--out", default="output", help="输出目录（默认 output）")
    args = parser.parse_args()

    cfg = load_config_cli()
    run_cfg = cfg["run"]
    keyword = args.keyword
    if not keyword:
        keyword = input("请输入要搜索的关键词（歌手名/歌名）：").strip()
    if not keyword:
        print("未输入关键词，退出。")
        sys.exit(1)

    limit = args.limit or run_cfg["max_songs"]
    pages = args.pages or run_cfg["comment_pages"]
    hot_limit = run_cfg["hot_limit"]

    result = run_pipeline(
        cfg=cfg,
        keyword=keyword,
        limit=limit,
        pages=pages,
        hot_limit=hot_limit,
        out=args.out,
        force_analyze=True,
        log=print,
    )

    print()
    print("================ 情感分析汇总 ================")
    for song in result["songs"]:
        print_song_summary(song)
    print("=" * 60)
    print(f"已导出: {result['song_dir']}\\*.json")
    print(f"       {result['songs_csv']}")
    print(f"       {result['comments_csv']}")


if __name__ == "__main__":
    main()