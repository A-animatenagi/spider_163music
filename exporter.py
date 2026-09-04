import csv
import json
import os

SONG_FIELDS = ["song_id", "name", "artists", "album", "album_id", "publish_time",
               "duration_ms", "lyric", "comments", "sentiment_summary"]
COMMENT_FIELDS = ["song_id", "song_name", "comment_id", "is_hot", "nickname",
                  "content", "liked_count", "time_str", "ip_location",
                  "sentiment", "sentiment_score", "sentiment_reason"]


def export_song_json(song, path):
    payload = {k: song.get(k) for k in SONG_FIELDS}
    payload["artists"] = song.get("artists") or []
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def export_csvs(songs, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    songs_path = os.path.join(out_dir, "songs.csv")
    comments_path = os.path.join(out_dir, "comments.csv")

    with open(songs_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SONG_FIELDS)
        writer.writeheader()
        for song in songs:
            row = {k: song.get(k) for k in SONG_FIELDS}
            row["artists"] = "/".join(song.get("artists") or [])
            row["sentiment_summary"] = json.dumps(song.get("sentiment_summary"), ensure_ascii=False)
            writer.writerow(row)

    with open(comments_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COMMENT_FIELDS)
        writer.writeheader()
        for song in songs:
            song_id = song.get("song_id")
            song_name = song.get("name")
            for c in song.get("comments") or []:
                writer.writerow({
                    "song_id": song_id,
                    "song_name": song_name,
                    "comment_id": c.get("comment_id"),
                    "is_hot": c.get("is_hot"),
                    "nickname": c.get("nickname"),
                    "content": c.get("content"),
                    "liked_count": c.get("liked_count"),
                    "time_str": c.get("time_str"),
                    "ip_location": c.get("ip_location"),
                    "sentiment": c.get("sentiment"),
                    "sentiment_score": c.get("sentiment_score"),
                    "sentiment_reason": c.get("sentiment_reason"),
                })
    return songs_path, comments_path