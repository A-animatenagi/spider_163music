"""本地情感分析（不依赖大模型 API）。

基于 SnowNLP 内置中文情感模型 + 音乐评论领域词典（含否定词翻转、程度副词加权）
混合打分，输出与大模型版完全一致的结果结构（{sentiment, score, reason}），
通过 config.json 中 sentiment.engine = "local" 切换启用，可视化与导出无需任何改动。

说明：SnowNLP 情感模型基于商品评论语料训练，直接用于音乐评论存在偏差
（如"太好听了"可能被判为负面），因此叠加领域词典进行修正：
词典有命中时按自适应权重（命中越多越信任词典）与 SnowNLP 得分融合，
无命中则直接使用 SnowNLP 得分。

准确率增强机制：
1. 全文逐处匹配词典（非仅首处），否定/程度上下文按每次出现独立判断；
2. 最长词优先并消耗区间，避免"单曲循环/循环"、"好听哭/好听"重复计分；
3. 转折子句（但/但是/可是/却/不过/然而 引导）权重 ×1.5，语义重心在后句；
4. 表情符号（👍❤🤮 等）计入正负向得分；
5. 纯水军标记（打卡/路过/沙发等）且无正负向命中时直接拉回中性，
   不让 SnowNLP 的随机得分主导；
6. 词典融合权重随命中数自适应：DICT_WEIGHT 起步，最高 DICT_WEIGHT_MAX；
7. 无词典/表情信号时 SnowNLP 视为语料外低置信读数，得分向中性收缩
   （NO_SIGNAL_DAMP），避免怀旧/中性句被低分误判为负面。
"""

import re

try:
    from snownlp import SnowNLP
    HAS_SNOWNLP = True
except Exception:
    HAS_SNOWNLP = False

from sentiment import SentimentAnalyzer

# 词典打分在混合得分中的基础权重（命中越多自适应上调，上限 DICT_WEIGHT_MAX）
DICT_WEIGHT = 0.6
DICT_WEIGHT_MAX = 0.85

# 无词典/表情信号时 SnowNLP 属语料外低置信读数，向中性收缩的阻尼系数：
# score = 0.5 + (base - 0.5) * NO_SIGNAL_DAMP，仅采信置信度高的极端读数，
# 避免怀旧/中性句（如“居然四年了”）被低分误判为负面
NO_SIGNAL_DAMP = 0.4

NEGATE_CHARS = set("不没别莫未无勿甭")
# 否定词误报豁免：这些词虽含否定字但本身不表否定（如“莫名”的“莫”），
# 计入否定窗口前先行剔除，避免把“莫名感到悲伤”的“悲伤”错误翻转为正向
NEGATE_EXCEPTIONS = ("莫名",)
DEGREE_WORDS = ("很", "太", "真", "超", "极", "特别", "非常", "最", "巨", "贼", "相当", "好", "死", "爆")

# 子句分隔与转折连词：转折后的子句是语义重心，加权 1.5
CLAUSE_RE = re.compile(r"[。！？!?；;，,、\n]+")
ADVERSATIVES = ("但", "但是", "可是", "却", "不过", "然而")

# 表情符号情感权重
POS_EMOJI = {"😍": 1.0, "🥰": 1.0, "❤": 0.8, "💕": 0.8, "👍": 0.8, "🔥": 0.6, "🎉": 0.6, "😭": 0.5, "💪": 0.5, "😊": 0.6, "😄": 0.6, "😀": 0.6}
NEG_EMOJI = {"👎": 1.0, "🤮": 1.0, "💩": 1.0, "😡": 0.8, "🤢": 0.8}

# 非情感水军标记：命中且无正负向词时直接判中性，不用 SnowNLP 随机得分
NEU_WORDS = ("打卡", "路过", "沙发", "前排", "日报", "报到", "占楼", "抢楼")

# 音乐评论常见正向词（权重约 0.2~1.2）
POS_WORDS = {
    "好听": 1.0, "喜欢": 1.0, "爱了": 1.0, "感动": 1.0, "神曲": 1.2, "惊艳": 1.0,
    "单曲循环": 1.0, "循环": 0.5, "经典": 0.8, "治愈": 1.0, "温暖": 0.8, "泪目": 0.8,
    "听哭": 0.9, "好听哭": 1.2, "封神": 1.0, "天籁": 1.0, "宝藏": 0.8, "共鸣": 0.7,
    "心声": 0.5, "回忆": 0.4, "青春": 0.4, "支持": 0.6, "永远的神": 1.2, "天花板": 1.0,
    "唱功": 0.5, "高级": 0.5, "舒服": 0.7, "开心": 0.8, "快乐": 0.8, "幸福": 0.8,
    "耐听": 0.9, "百听不厌": 1.2, "推荐": 0.5, "情怀": 0.4, "爷青回": 1.0,
    "还行": 0.4, "还可以": 0.4, "上头": 0.8, "绝了": 1.0, "入坑": 0.6, "赞": 0.6,
}

# 音乐评论常见负向词
NEG_WORDS = {
    "难听": 1.0, "垃圾": 1.2, "失望": 0.9, "浪费": 0.8, "刺耳": 1.0, "嘈杂": 0.6,
    "抄袭": 0.9, "口水歌": 0.8, "烂": 1.0, "差劲": 1.0, "恶心": 1.1, "讨厌": 0.9,
    "翻车": 0.9, "拉胯": 0.9, "无聊": 0.6, "做作": 0.8, "难崩": 0.7, "不行": 0.6,
    "悲伤": 0.5, "难过": 0.5, "遗憾": 0.4, "可惜": 0.4, "一般": 0.4,
    "无语": 0.6, "尴尬": 0.6, "退钱": 0.9,
}

# 最长词优先排序，保证子串重叠时只计最长词（如"单曲循环"优先于"循环"）
_LEX_ENTRIES = sorted(
    [(w, v, True) for w, v in POS_WORDS.items()] + [(w, v, False) for w, v in NEG_WORDS.items()],
    key=lambda t: -len(t[0]))


class LocalSentimentAnalyzer:
    engine = "local"

    def __init__(self, neg_threshold=0.4, pos_threshold=0.6):
        self.neg_threshold = min(max(float(neg_threshold or 0.4), 0.0), 1.0)
        self.pos_threshold = min(max(float(pos_threshold or 0.6), self.neg_threshold), 1.0)
        self.model = "snownlp+词典"
        self.enabled = HAS_SNOWNLP

    @property
    def base_url(self):
        return ""

    @property
    def label(self):
        return "本地 SnowNLP + 词典模型（无需 API）"

    @staticmethod
    def _clause_spans(text):
        spans, start = [], 0
        for m in CLAUSE_RE.finditer(text):
            spans.append((start, m.start()))
            start = m.end()
        spans.append((start, len(text)))
        return spans

    @staticmethod
    def _clause_weight(text, spans, idx):
        """idx 所在子句的权重：转折连词引导的子句是语义重心，加权 1.5。"""
        for s, e in spans:
            if s <= idx < e:
                return 1.5 if text[s:e].lstrip().startswith(ADVERSATIVES) else 1.0
        return 1.0

    @classmethod
    def _lexicon(cls, text):
        """扫描领域词典与表情，返回 (正向总分, 负向总分, 正向命中词, 负向命中词)。

        全文逐处匹配（非仅首处）；最长词优先并消耗区间避免重复计分；
        否定词窗口为命中前 4 字，奇数个否定词才翻转。
        """
        spans = cls._clause_spans(text)
        consumed = []
        pos_total = neg_total = 0.0
        pos_hits, neg_hits = [], []
        for word, weight, is_pos in _LEX_ENTRIES:
            for m in re.finditer(re.escape(word), text):
                s, e = m.span()
                if any(s < ce and cs < e for cs, ce in consumed):
                    continue
                consumed.append((s, e))
                pre = text[max(0, s - 4):s]
                for ex in NEGATE_EXCEPTIONS:
                    pre = pre.replace(ex, "")
                negated = sum(pre.count(ch) for ch in NEGATE_CHARS) % 2 == 1
                w = weight
                if not negated and any(d in pre for d in DEGREE_WORDS):
                    w *= 1.3
                w *= cls._clause_weight(text, spans, s)
                if is_pos ^ negated:
                    pos_total += w
                    pos_hits.append(word)
                else:
                    neg_total += w
                    neg_hits.append(word)
        for emo, w in POS_EMOJI.items():
            cnt = text.count(emo)
            if cnt:
                pos_total += w * cnt
                pos_hits.append(emo)
        for emo, w in NEG_EMOJI.items():
            cnt = text.count(emo)
            if cnt:
                neg_total += w * cnt
                neg_hits.append(emo)
        return pos_total, neg_total, pos_hits, neg_hits

    def _score(self, text):
        """混合打分：词典命中时按权重融合，否则仅用 SnowNLP。返回 (score, 说明)。"""
        base = None
        if HAS_SNOWNLP:
            try:
                base = float(SnowNLP(text).sentiments)
            except Exception:
                base = None
        pos_total, neg_total, pos_hits, neg_hits = self._lexicon(text)
        detail = ""
        if pos_total or neg_total:
            dict_score = pos_total / (pos_total + neg_total)
            if base is not None:
                # 命中越多越信任词典：权重自 DICT_WEIGHT 自适应上调至 DICT_WEIGHT_MAX
                dw = min(DICT_WEIGHT + 0.1 * (len(pos_hits) + len(neg_hits) - 1), DICT_WEIGHT_MAX)
                score = dw * dict_score + (1 - dw) * base
            else:
                score = dict_score
            hits = ", ".join(pos_hits + ["~" + w for w in neg_hits]) or "无"
            detail = f"，词典命中[{hits}]，SnowNLP {base:.2f}" if base is not None \
                else f"，词典命中[{hits}]（SnowNLP 不可用）"
        elif any(w in text for w in NEU_WORDS):
            # 纯水军/报到类评论无情感倾向，直接拉回中性，避免 SnowNLP 随机得分主导
            score = 0.5
            detail = "，命中水军标记，拉回中性"
        else:
            if base is None:
                return None, ""
            # 无任何词典/表情信号时 SnowNLP 为语料外低置信读数，向中性收缩，
            # 只采信置信度高的极端值，避免怀旧/中性句被误判为负面/正面
            score = 0.5 + (base - 0.5) * NO_SIGNAL_DAMP
            detail = "，无词典信号，SnowNLP 向中性收缩"
        return max(0.0, min(1.0, score)), detail

    def analyze_one(self, comment):
        text = (comment.get("content") or "").strip()
        if not text:
            return {"sentiment": "unknown", "score": None, "reason": "空评论"}
        score, detail = self._score(text)
        if score is None:
            return {"sentiment": "unknown", "score": None, "reason": "本地分析失败",
                    "error": "SnowNLP 不可用且词典无命中"}
        score = round(score, 3)
        if score >= self.pos_threshold:
            sentiment, why = "positive", f"得分 {score} ≥ {self.pos_threshold}，判定为正面"
        elif score <= self.neg_threshold:
            sentiment, why = "negative", f"得分 {score} ≤ {self.neg_threshold}，判定为负面"
        else:
            sentiment, why = "neutral", f"得分 {score} 介于 {self.neg_threshold}~{self.pos_threshold}，判定为中性"
        return {"sentiment": sentiment, "score": score, "reason": f"本地模型：{why}{detail}"}

    def analyze_many(self, comments, batch_size=None):
        # 纯 CPU 计算且极快，无需并发/分批
        return [self.analyze_one(c) for c in comments]

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
            SentimentAnalyzer._attach_summary(song)
        return songs
