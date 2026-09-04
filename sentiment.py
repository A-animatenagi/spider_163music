import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

SYSTEM_PROMPT = (
    "你是一名专业的网易云音乐歌曲评论情感分析助手。"
    "对用户给出的单条评论进行情感分析，只输出一个JSON对象，不要输出任何其他文字或代码块标记。"
    'JSON格式：{"sentiment": "positive", "score": 0.9, "reason": "一句话简述"}。'
    '其中 sentiment 取值只能是 "positive"（正面）、"neutral"（中性）、"negative"（负面）三者之一；'
    'score 是0到1之间的小数，0代表非常负面，0.5代表中性，1代表非常正面。'
)

BATCH_SYSTEM = (
    "你是一名专业的网易云音乐歌曲评论情感分析助手。"
    "对下列多条评论逐条进行情感分析，只输出一个JSON对象，不要输出任何其他文字或代码块标记。"
    'JSON格式：{"results": [{"index": 0, "sentiment": "positive", "score": 0.9, "reason": "一句话"}, ...]}。'
    "其中 index 必须与输入编号一一对应且不遗漏任何一条，results 的数量必须等于输入的评论数量。"
    'sentiment 取值只能是 "positive"（正面）、"neutral"（中性）、"negative"（负面）三者之一；'
    "score 是0到1之间的小数，0代表非常负面，0.5代表中性，1代表非常正面。"
)


def extract_json(text):
    if not text:
        return None
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


SENTIMENT_LABELS = {
    "positive": "positive", "pos": "positive", "正面": "positive", "积极": "positive",
    "正向": "positive", "正": "positive", "满意": "positive", "喜欢": "positive",
    "neutral": "neutral", "neu": "neutral", "中性": "neutral", "中立": "neutral",
    "中": "neutral", "平淡": "neutral", "一般": "neutral", "普通": "neutral",
    "平常": "neutral", "客观": "neutral", "无感": "neutral", "平和": "neutral",
    "negative": "negative", "neg": "negative", "负面": "negative", "消极": "negative",
    "负向": "negative", "负": "negative", "反感": "negative", "悲伤": "negative",
    "unknown": "unknown", "未知": "unknown", "无法判断": "unknown",
}

SENTIMENT_KEYS = ["sentiment", "emotion", "label", "type", "情感", "情绪", "标签", "态度"]


def _norm_sentiment(value):
    if value is None:
        return None
    key = str(value).strip().lower().replace(" ", "").replace("_", "")
    return SENTIMENT_LABELS.get(key)


def _norm_score(value):
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        score = float(value)
    else:
        try:
            score = float(str(value).strip().rstrip("%"))
        except (TypeError, ValueError):
            return None
    if score > 1.0 and score <= 100.0:
        score = score / 100.0
    return max(0.0, min(1.0, score))


def parse_result(content):
    obj = extract_json(content)
    if not obj or not isinstance(obj, dict):
        return {"sentiment": "unknown", "score": None, "reason": "解析失败", "error": "模型输出无法解析"}
    sentiment = None
    for key in SENTIMENT_KEYS:
        if key in obj:
            sentiment = _norm_sentiment(obj.get(key))
            if sentiment:
                break
    if not sentiment:
        return {"sentiment": "unknown", "score": None, "reason": str(content)[:200], "error": "sentiment取值非法"}
    score = _norm_score(obj.get("score"))
    return {
        "sentiment": sentiment,
        "score": score if sentiment != "unknown" else None,
        "reason": str(obj.get("reason") or obj.get("原因") or obj.get("解释") or ""),
    }


def parse_batch(content):
    obj = extract_json(content)
    if not obj:
        return None
    raw = obj.get("results") if isinstance(obj, dict) else obj
    if isinstance(obj, dict) and isinstance(raw, dict):
        items = []
        for k in sorted(raw.keys(), key=lambda x: int(x) if str(x).isdigit() else 0):
            v = raw[k]
            if isinstance(v, dict):
                items.append({"index": int(k) if str(k).isdigit() else k, **v})
            else:
                items.append({"index": int(k) if str(k).isdigit() else k, "sentiment": v})
        raw = items
    if not isinstance(raw, list):
        return None
    mapping = {}
    for item in raw:
        if not isinstance(item, dict):
            continue
        idx = item.get("index", item.get("id", item.get("序号")))
        if idx is None:
            continue
        if isinstance(idx, str) and idx.strip().isdigit():
            idx = int(idx)
        mapping[int(idx)] = item
    return mapping


class SentimentAnalyzer:
    engine = "llm"

    def __init__(self, base_url, api_key, model, timeout=60, concurrency=4, batch_size=12):
        self.base_url = (base_url or "").rstrip("/")
        self.api_key = api_key or ""
        self.model = model or "deepseek-chat"
        self.timeout = timeout
        self.concurrency = max(1, concurrency)
        self.batch_size = max(1, batch_size or 12)
        self.enabled = bool(
            self.api_key
            and self.api_key.strip() not in ("sk-你的APIKey", "sk-xxxx", "your-api-key")
            and len(self.api_key.strip()) >= 8
        )

    @property
    def label(self):
        return f"大模型 {self.model}，并发 {self.concurrency}"

    @property
    def endpoint(self):
        return self.base_url + "/chat/completions"

    def _post(self, payload):
        headers = {
            "Authorization": "Bearer " + self.api_key,
            "Content-Type": "application/json",
        }
        resp = requests.post(self.endpoint, json=payload, headers=headers, timeout=self.timeout)
        resp.raise_for_status()
        return (resp.json()["choices"][0]["message"]["content"] or "")

    def analyze_one(self, comment):
        text = (comment.get("content") or "").strip()
        if not text:
            return {"sentiment": "unknown", "score": None, "reason": "空评论"}
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "请分析这条网易云音乐歌曲评论的情感：\n" + text},
            ],
            "temperature": 0.2,
        }
        try:
            return parse_result(self._post(payload))
        except Exception as e:
            return {"sentiment": "unknown", "score": None, "reason": str(e), "error": "调用失败"}

    def _call_batch(self, indexes, texts):
        if not indexes:
            return {}
        lines = [f"[{i}] {t}" for i, t in zip(indexes, texts)]
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": BATCH_SYSTEM},
                {"role": "user", "content": "请对下面每条评论进行情感分析，index 从 0 开始编号：\n" + "\n".join(lines)},
            ],
            "temperature": 0.2,
        }
        content = self._post(payload)
        return parse_batch(content) or {}

    def analyze_many(self, comments, batch_size=None):
        batch_size = batch_size or self.batch_size
        n = len(comments)
        texts = [(c.get("content") or "").strip() for c in comments]
        empty_res = {"sentiment": "unknown", "score": None, "reason": "空评论"}
        results = [empty_res if not t else None for t in texts]
        indexes = [i for i, t in enumerate(texts) if t]

        batches = [indexes[i:i + batch_size] for i in range(0, len(indexes), batch_size)]

        def work(batch_idxs):
            mapping = self._call_batch(batch_idxs, [texts[i] for i in batch_idxs])
            return batch_idxs, mapping

        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            futures = [pool.submit(work, b) for b in batches]
            for fut in as_completed(futures):
                batch_idxs, mapping = fut.result()
                for i in batch_idxs:
                    item = mapping.get(i)
                    if item is None:
                        continue
                    r = parse_result(json.dumps(item, ensure_ascii=False))
                    if r.get("sentiment") != "unknown":
                        results[i] = r

        # 对批量里未成功/解析失败的评论，回退到逐条分析
        missing = [i for i in indexes if results[i] is None]
        if missing:
            with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
                futures = {pool.submit(self.analyze_one, comments[i]): i for i in missing}
                for fut in as_completed(futures):
                    results[futures[fut]] = fut.result()

        for i, r in enumerate(results):
            if r is None:
                results[i] = {"sentiment": "unknown", "score": None, "reason": "解析失败", "error": "批量解析失败"}
        return results

    def analyze_songs(self, songs):
        for song in songs:
            comments = song.get("comments") or []
            if not comments:
                song["sentiment_summary"] = {"total": 0, "positive": 0, "neutral": 0,
                                             "negative": 0, "unknown": 0, "avg_score": None}
                continue
            results = self.analyze_many(comments)
            for c, r in zip(comments, results):
                c["sentiment"] = r.get("sentiment")
                c["sentiment_score"] = r.get("score")
                c["sentiment_reason"] = r.get("reason")
            self._attach_summary(song)
        return songs

    @staticmethod
    def _attach_summary(song):
        comments = song.get("comments") or []
        summary = {"total": len(comments), "positive": 0, "neutral": 0, "negative": 0,
                   "unknown": 0, "avg_score": None}
        scores = []
        for c in comments:
            s = c.get("sentiment")
            if s in summary:
                summary[s] += 1
            score = c.get("sentiment_score")
            if score is not None:
                scores.append(score)
        if scores:
            summary["avg_score"] = round(sum(scores) / len(scores), 3)
        song["sentiment_summary"] = summary