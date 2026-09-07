import base64
import json
import random
import string
import time

import requests
from Crypto.Cipher import AES

PUBLIC_KEY = "010001"
MODULUS = (
    "00e0b509f6259df8642dbc35662901477df22677ec152b5ff68ace615bb7b7251"
    "52b3ab17a876aea8a5aa76d2e417629ec4ee341f56135fccf695280104e0312e"
    "cbda92557c93870114af6c9d05c4f7f0c3685b7a46bee255932575cce10b424d"
    "813cfe4875d3e82047b97ddef52741d546b8e289dc6935b3ece0462db0a22b8e7"
)
IV = "0102030405060708"
PRESET_KEY = "0CoJUm6Qyw8W8jud"
BASE62 = string.digits + string.ascii_lowercase

API_SEARCH = "https://music.163.com/api/cloudsearch/pc"
API_SONG_DETAIL = "https://music.163.com/api/song/detail/"
API_LYRIC = "https://music.163.com/api/song/lyric"
API_COMMENTS = "https://music.163.com/weapi/comment/resource/comments/get"
API_HOT_COMMENTS = "https://music.163.com/weapi/v1/resource/hotcomments/"

DEFAULT_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)


def aes_encrypt(text, key):
    data = json.dumps(text, separators=(",", ":")) if not isinstance(text, str) else text
    pad = 16 - len(data) % 16
    data += chr(pad) * pad
    cipher = AES.new(key.encode(), AES.MODE_CBC, IV.encode())
    return base64.b64encode(cipher.encrypt(data.encode())).decode()


def rsa_encrypt(text):
    reverse = text[::-1]
    return pow(
        int(reverse.encode().hex(), 16), int(PUBLIC_KEY, 16), int(MODULUS, 16)
    ).__format__("x").zfill(256)


def weapi(payload):
    sec_key = "".join(random.choice(BASE62) for _ in range(16))
    params = aes_encrypt(aes_encrypt(payload, PRESET_KEY), sec_key)
    return {"params": params, "encSecKey": rsa_encrypt(sec_key)}


class NeteaseCrawler:
    def __init__(self, user_agent=None, sleep=1.0, timeout=15, max_retries=3):
        self.user_agent = user_agent or DEFAULT_UA
        self.sleep = sleep
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers.update(self._base_headers())

    def _base_headers(self):
        return {
            "User-Agent": self.user_agent,
            "Referer": "https://music.163.com/",
            "Cookie": "os=pc; appver=2.9.7",
        }

    def _pause(self):
        if self.sleep > 0:
            time.sleep(self.sleep)

    def _get(self, url, params=None):
        last_err = None
        for _ in range(self.max_retries):
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)
                resp.raise_for_status()
                return resp.json()
            except Exception as e:
                last_err = e
                self._pause()
        raise RuntimeError(f"GET {url} failed: {last_err}")

    def _weapi_post(self, url, payload):
        last_err = None
        headers = dict(self._base_headers())
        headers["Content-Type"] = "application/x-www-form-urlencoded"
        for _ in range(self.max_retries):
            try:
                resp = self.session.post(
                    url, data=weapi(payload), headers=headers, timeout=self.timeout
                )
                resp.raise_for_status()
                data = resp.json()
                if data.get("code") != 200:
                    raise RuntimeError(f"weapi error code={data.get('code')} msg={data.get('msg')}")
                return data
            except Exception as e:
                last_err = e
                self._pause()
        raise RuntimeError(f"POST {url} failed: {last_err}")

    def search_songs(self, keyword, limit=10):
        data = self._get(API_SEARCH, params={"s": keyword, "type": 1, "limit": limit, "offset": 0})
        songs = (data.get("result") or {}).get("songs") or []
        result = []
        for s in songs:
            result.append({
                "song_id": s.get("id"),
                "name": s.get("name"),
                "artists": [a.get("name") for a in (s.get("ar") or [])],
                "album": (s.get("al") or {}).get("name"),
                "duration_ms": s.get("dt"),
                "publish_time": s.get("publishTime"),
                "song_id_raw": s.get("id"),
            })
        return result

    def get_song_detail(self, song_id):
        data = self._get(API_SONG_DETAIL, params={"ids": "[%d]" % song_id})
        songs = data.get("songs") or []
        if not songs:
            return None
        s = songs[0]
        album = s.get("album") or {}
        return {
            "song_id": s.get("id"),
            "name": s.get("name"),
            "artists": [a.get("name") for a in (s.get("artists") or [])],
            "album": album.get("name"),
            "album_id": album.get("id"),
            "album_pic": album.get("picUrl"),
            "publish_time": album.get("publishTime"),
            "duration_ms": s.get("duration"),
            "comment_count": s.get("commentThreadId"),
        }

    def get_lyric_full(self, song_id):
        """返回结构化歌词：原文 / 翻译 / 罗马音，均带 LRC 时间轴，字段可能为空。"""
        data = self._get(API_LYRIC, params={
            "id": song_id, "lv": -1, "kv": -1, "tv": -1, "rv": -1})
        return {
            "lyric": (data.get("lrc") or {}).get("lyric") or "",
            "trans": (data.get("tlyric") or {}).get("lyric") or "",
            "roma": (data.get("romalrc") or {}).get("lyric") or "",
        }

    def get_lyric(self, song_id):
        return self.get_lyric_full(song_id)["lyric"]

    @staticmethod
    def _normalize_comment(c):
        user = c.get("user") or {}
        ip = c.get("ipLocation") or {}
        return {
            "comment_id": c.get("commentId"),
            "content": c.get("content"),
            "user_id": user.get("userId"),
            "nickname": user.get("nickname"),
            "avatar": user.get("avatarUrl"),
            "time": c.get("time"),
            "time_str": c.get("timeStr"),
            "liked_count": c.get("likedCount"),
            "ip_location": ip.get("location"),
        }

    def get_hot_comments(self, song_id, limit=20):
        rid = "R_SO_4_%d" % song_id
        url = API_HOT_COMMENTS + rid
        data = self._weapi_post(url, {"rid": rid, "offset": 0, "limit": limit, "csrf_token": ""})
        items = data.get("data") or []
        return [self._normalize_comment(c) for c in items]

    def get_comments(self, song_id, pages=1):
        rid = "R_SO_4_%d" % song_id
        collected, seen, cursor, page_no = [], set(), "-1", 1
        for _ in range(max(pages, 1)):
            payload = {
                "rid": rid,
                "threadId": rid,
                "pageNo": str(page_no),
                "pageSize": "20",
                "cursor": cursor,
                "offset": "0",
                "orderType": "1",
                "csrf_token": "",
            }
            data = self._weapi_post(API_COMMENTS, payload)
            items = (data.get("data") or {}).get("comments") or []
            for c in items:
                norm = self._normalize_comment(c)
                if norm["comment_id"] not in seen:
                    seen.add(norm["comment_id"])
                    collected.append(norm)
            cursor = (data.get("data") or {}).get("cursor") or cursor
            if not items:
                break
            page_no += 1
        return collected

    def get_song_full(self, song_id, comment_pages=3, hot_limit=20):
        detail = self.get_song_detail(song_id)
        if not detail:
            return None
        lyric = self.get_lyric(song_id)
        hot = self.get_hot_comments(song_id, limit=hot_limit)
        normal = self.get_comments(song_id, pages=comment_pages)
        comments = []
        seen = set()
        for c in hot + normal:
            if c["comment_id"] not in seen:
                seen.add(c["comment_id"])
                c = dict(c)
                c["is_hot"] = True
                comments.append(c)
        return {
            "song_id": detail["song_id"],
            "name": detail["name"],
            "artists": detail["artists"],
            "album": detail["album"],
            "album_id": detail["album_id"],
            "album_pic": detail["album_pic"],
            "publish_time": detail["publish_time"],
            "duration_ms": detail["duration_ms"],
            "lyric": lyric,
            "comments": comments,
        }