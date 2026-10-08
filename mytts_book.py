#!/usr/bin/env python3
"""MyTTS · подготовка книги к озвучке голосами персонажей.

Одна команда — EPUB на входе, файл .mytts-book на выходе (импорт в MyTTS: настройки LLM → мультиголос →
«Книги с голосами персонажей» → «Загрузить файл книги»):

  python mytts_book.py process book.epub                      # Ollama Cloud, ключ OLLAMA_API_KEY
  python mytts_book.py process book.epub --provider deepseek  # API DeepSeek, ключ DEEPSEEK_API_KEY
  python mytts_book.py process book.epub --provider groq      # Groq (Qwen 3.8 27B), ключ GROQ_API_KEY

Скрипт сам находит имена, фамилии, отчества, прозвища и обращения (морфология pymorphy3 и статистика книги),
LLM только сопоставляет их: какие формы — один персонаж. Сомнительное уходит в «прочие». Роман или сборник
рассказов определяется по оглавлению. В файл .mytts-book попадают персонажи и отпечатки предложений, текста
книги в нём нет. Отдельные шаги: extract → llm → apply → voices → export (см. README.md).
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import hashlib
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from argparse import Namespace
import zipfile
from dataclasses import dataclass, field
from pathlib import PurePosixPath

try:
    import pymorphy3
    from bs4 import BeautifulSoup
except ImportError:  # pragma: no cover
    sys.exit("Нужны pymorphy3 и beautifulsoup4: pip install pymorphy3 pymorphy3-dicts-ru beautifulsoup4")

ACUTE = "́"
WORD = re.compile(r"[А-ЯЁа-яё]+(?:-[А-ЯЁа-яё]+)*")
SENTENCE_END = ".!?…"
OPENERS = " \t«\"„“(—–-'"
DASH = re.compile(r"\s[—–]\s")
BLOCKS = ["p", "h1", "h2", "h3", "h4", "h5", "h6", "li"]
# Обращения, которые сами указывают на персонажа («сказал князь») или стоят перед именем («генерал Иволгин»).
TITLES = {
    "князь", "княгиня", "княжна", "граф", "графиня", "барон", "баронесса", "генерал", "генеральша",
    "полковник", "капитан", "поручик", "подпоручик", "майор", "господин", "госпожа", "мадам", "мадемуазель",
    "мсье", "сударь", "сударыня", "барыня", "барин", "доктор", "профессор", "чиновник", "купец", "купчиха",
}
NOT_PEOPLE = {"бог", "господь", "господи", "христос", "богородица", "аллах", "сатана", "иисус"}
CASE_ENDINGS = ("ами", "ому", "ему", "ыми", "ими", "ой", "ым", "им", "ом", "ем", "ых", "их", "а", "у", "е", "ы", "и", "ю", "я")
CASES = ("nomn", "gent", "datv", "accs", "ablt", "loct", "voct", "gen2", "acc2", "loc2")


# ---------------------------------------------------------------- книга

@dataclass
class Book:
    title: str
    author: str
    sections: list            # [{"id": "s1", "title": "Старшая сестра"}]
    paragraphs: list          # [(номер раздела, текст)]


def read_epub(path: str) -> Book:
    """Абзацы в порядке чтения, разделы — верхний уровень оглавления (вложенные пункты — части рассказа)."""
    with zipfile.ZipFile(path) as z:
        container = ET.fromstring(z.read("META-INF/container.xml"))
        opf_path = next(e.get("full-path") for e in container.iter() if e.tag.endswith("rootfile"))
        opf = ET.fromstring(z.read(opf_path))
        base = PurePosixPath(opf_path).parent
        title = next((e.text for e in opf.iter() if e.tag.endswith("}title") and e.text), PurePosixPath(path).stem)
        author = next((e.text for e in opf.iter() if e.tag.endswith("}creator") and e.text), "")
        items = {e.get("id"): e for e in opf.iter() if e.tag.endswith("}item")}
        spine = [items[e.get("idref")].get("href") for e in opf.iter() if e.tag.endswith("}itemref") and e.get("idref") in items]
        toc = top_level_toc(z, base, items, opf)
        # Пункт оглавления: файл → [(якорь или None, номер раздела)]
        starts: dict[str, list] = collections.defaultdict(list)
        sections = []
        for label, href in toc:
            file, _, anchor = href.partition("#")
            starts[normalize_href(file)].append((anchor or None, len(sections)))
            sections.append({"id": f"s{len(sections) + 1}", "title": label})
        if not sections:  # без оглавления: раздел = файл
            for href in spine:
                starts[normalize_href(href)].append((None, len(sections)))
                sections.append({"id": f"s{len(sections) + 1}", "title": PurePosixPath(href).stem})
        paragraphs: list[tuple[int, str]] = []
        current = 0
        for href in spine:
            if not href.endswith((".xhtml", ".html", ".htm")):
                continue
            pending = dict(starts.get(normalize_href(href), []))
            if None in pending:
                current = pending.pop(None)
            soup = BeautifulSoup(z.read(str(base / href)).decode("utf-8", "replace"), "html.parser")
            for tag in soup.find_all(True):
                if tag.get("id") in pending:
                    current = pending.pop(tag.get("id"))
                if tag.name in BLOCKS and not tag.find(BLOCKS):
                    text = clean(tag.get_text(" "))
                    if text:
                        paragraphs.append((current, text))
    used = sorted({s for s, _ in paragraphs})
    renumber = {old: new for new, old in enumerate(used)}
    sections = [dict(sections[old], id=f"s{renumber[old] + 1}") for old in used]
    return Book(title, clean(author), sections, [(renumber[s], t) for s, t in paragraphs])


def top_level_toc(z, base, items, opf) -> list[tuple[str, str]]:
    nav = next((e for e in items.values() if "nav" in (e.get("properties") or "").split()), None)
    if nav is not None:
        soup = BeautifulSoup(z.read(str(base / nav.get("href"))).decode("utf-8", "replace"), "html.parser")
        toc = soup.find("nav", attrs={"epub:type": "toc"}) or soup.find("nav")
        ol = toc.find("ol") if toc else None
        if ol:
            out = []
            for li in ol.find_all("li", recursive=False):
                a = li.find("a")
                if a and a.get("href"):
                    out.append((clean(a.get_text(" ")), resolve(nav.get("href"), a.get("href"))))
            if out:
                return out
    spine = next((e for e in opf.iter() if e.tag.endswith("}spine")), None)
    ncx_id = spine.get("toc") if spine is not None else None
    ncx = items.get(ncx_id) if ncx_id else next((e for e in items.values() if e.get("href", "").endswith(".ncx")), None)
    if ncx is None:
        return []
    root = ET.fromstring(z.read(str(base / ncx.get("href"))))
    nav_map = next((e for e in root.iter() if e.tag.endswith("}navMap")), None)
    out = []
    for point in (nav_map or []):
        if not point.tag.endswith("}navPoint"):
            continue
        label = next((e.text for e in point.iter() if e.tag.endswith("}text") and e.text), "")
        content = next((e for e in point if e.tag.endswith("}content")), None)
        if content is not None:
            out.append((clean(label), resolve(ncx.get("href"), content.get("src"))))
    return out


def resolve(origin: str, href: str) -> str:
    folder = PurePosixPath(origin).parent
    return str(folder / href) if str(folder) != "." else href


def normalize_href(href: str) -> str:
    return os.path.normpath(href.split("#")[0]).replace("\\", "/")


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\xa0", " ").replace(ACUTE, "")).strip()


# ---------------------------------------------------------------- отпечатки (формат .mytts-book v1)
# Одинаково в приложении (Kotlin, books/BookFingerprint.kt): предложения по SENTENCES, буквы и цифры в нижнем
# регистре (ё → е, без ударений), первые 48; меньше 24 — не используется; FNV-1a 64 по кодам символов.

SENTENCES = re.compile(r"(?<=[.!?…])\s+")
FNV_OFFSET, FNV_PRIME, MASK64 = 0xcbf29ce484222325, 0x100000001b3, (1 << 64) - 1


def letters(text: str) -> str:
    return "".join(c for c in text.lower().replace("ё", "е").replace(ACUTE, "") if "а" <= c <= "я" or "a" <= c <= "z" or "0" <= c <= "9")


def sentence_fingerprints(text: str) -> list[str]:
    out = []
    for sentence in SENTENCES.split(text):
        key = letters(sentence)[:48]
        if len(key) < 24:
            continue
        h = FNV_OFFSET
        for c in key:
            h = ((h ^ ord(c)) * FNV_PRIME) & MASK64
        out.append(f"{h:016x}")
    return out


def fingerprint(text: str) -> str | None:
    """Первые 80 букв абзаца без регистра, ё, ударений и знаков: так же нормализует читалка."""
    letters = re.sub(r"[^a-zа-я0-9]", "", text.lower().replace("ё", "е").replace(ACUTE, ""))
    return hashlib.sha1(letters[:80].encode()).hexdigest()[:12] if len(letters) >= 24 else None


def sentence_start(text: str, start: int) -> bool:
    i = start - 1
    while i >= 0 and text[i] in OPENERS:
        i -= 1
    return i < 0 or text[i] in SENTENCE_END


# ---------------------------------------------------------------- кандидаты

@dataclass
class Candidate:
    key: str                      # «настасья филипповна»: слова в именительном падеже
    kind: str                     # name / title / family
    scope: int                    # раздел (сборник) или 0 (роман)
    count: int = 0
    speaker: int = 0              # назван в ремарке диалога в именительном («— сказал князь»)
    genders: collections.Counter = field(default_factory=collections.Counter)  # "verb:m", "Patr:f"…
    roles: collections.Counter = field(default_factory=collections.Counter)    # Name / Patr / Surn
    titles: collections.Counter = field(default_factory=collections.Counter)   # «князь» перед «Мышкин»
    forms: collections.Counter = field(default_factory=collections.Counter)    # как написано в книге
    sections: collections.Counter = field(default_factory=collections.Counter)
    examples: list = field(default_factory=list)
    together: collections.Counter = field(default_factory=collections.Counter)


@dataclass
class Mention:
    start: int
    end: int
    cand: Candidate
    nominative: bool


class Extractor:
    def __init__(self) -> None:
        self.morph = pymorphy3.MorphAnalyzer()
        self.cache: dict[str, list] = {}
        self.capitalized: collections.Counter = collections.Counter()
        self.lower: collections.Counter = collections.Counter()
        self.capital_inside: collections.Counter = collections.Counter()

    def named(self, word: str) -> list:
        """Разборы как имени/фамилии/отчества в единственном числе («Рогожину» — фамилия, не «рогожина»)."""
        if word not in self.cache:
            self.cache[word] = self.morph.parse(word)
        return [p for p in self.cache[word] if {"Name", "Surn", "Patr"} & set(p.tag.grammemes)]

    def first(self, word: str):
        if word not in self.cache:
            self.cache[word] = self.morph.parse(word)
        return self.cache[word][0]

    def statistics(self, paragraphs) -> None:
        for _, text in paragraphs:
            for m in WORD.finditer(text):
                w = m.group()
                k = w.lower().replace("ё", "е")
                if w[0].islower():
                    self.lower[k] += 1
                else:
                    self.capitalized[k] += 1
                    if not sentence_start(text, m.start()):
                        self.capital_inside[k] += 1

    def is_name(self, word: str, at_start: bool) -> bool:
        k = word.lower().replace("ё", "е")
        if not word[0].isupper() or (len(word) > 1 and word.isupper()) or k in NOT_PEOPLE or k in TITLES:
            return False
        parses = self.cache.get(word) or self.morph.parse(word)
        self.cache[word] = parses
        # «Фу», «Ай», «Ну» угадыватель разбирает и как имя: служебное слово или междометие именем не считаем.
        if any(p.tag.POS in ("INTJ", "PRCL", "CONJ", "PREP", "NPRO") for p in parses):
            return False
        tagged = any({"Name", "Surn", "Patr"} & set(p.tag.grammemes) for p in parses)
        if not tagged and {"Geox", "Orgn"} & set(parses[0].tag.grammemes):
            return False
        if {"ADJF", "Poss"} <= set(parses[0].tag.grammemes) or self.possessive(k):  # «Зинина», «Лидиной» — чьё-то
            return False
        if self.lower[k] > self.capital_inside[k]:  # нарицательное, просто в начале предложения
            return False
        if at_start:
            return self.capital_inside[k] > 0 or (tagged and self.lower[k] == 0)
        return tagged or self.capital_inside[k] >= 2

    POSSESSIVE = re.compile(r"^(.{2,}?)[иы]н(?:а|о|ы|ой|ою|ому|ым|ом|ых|ыми|у|е)?$")

    def possessive(self, k: str) -> bool:
        """«Зинина», «Анину», «Ксенин»: притяжательное от имени, которое в книге встречается намного чаще.
        Настоящая фамилия («Рогожин») остаётся: «Рогожа» в книге нет."""
        m = self.POSSESSIVE.match(k)
        if not m:
            return False
        owner = max(self.capitalized[m.group(1) + "а"], self.capitalized[m.group(1) + "я"])
        return owner >= max(5, 3 * self.capitalized[k])

    def chain_key(self, words: list[str]) -> tuple[str, list, bool, bool]:
        """Ключ цепочки «Евгения Павловича» → «евгений павлович»: падеж и род согласуются по всем словам.
        Возвращает (ключ, [(род, роль)], семья, именительный падеж)."""
        options = [[p for p in self.named(w) if "sing" in p.tag.grammemes or "Sgtm" in p.tag.grammemes] for w in words]
        family = any(self.named(w) and not opt for w, opt in zip(words, options))
        shared = None
        for opt in options:
            if opt:
                cases = {(p.tag.case, p.tag.gender) for p in opt}
                shared = cases if shared is None else shared & cases
        keys, info = [], []
        nominative = True
        for w, opt in zip(words, options):
            low = w.lower().replace("ё", "е")
            if not opt:
                parses = self.named(w)
                if parses:  # только множественное: «Епанчиных» — семья
                    keys.append(parses[0].normal_form.replace("ё", "е"))
                    info.append((None, next(r for r in ("Name", "Patr", "Surn") if r in parses[0].tag.grammemes)))
                else:  # нет в словаре: «Рогожина», «Фердыщенка» → форма, которая сама встречается в книге
                    keys.append(self.unknown_base(low))
                    info.append((None, None))
                    nominative = nominative and keys[-1] == low
                continue
            if shared:
                agreed = [p for p in opt if (p.tag.case, p.tag.gender) in shared]
                opt = agreed or opt
            forms: dict[str, list] = collections.defaultdict(list)
            for p in opt:
                inflected = p.inflect({"nomn", "sing"})
                forms[(inflected.word if inflected else p.normal_form).replace("ё", "е")].append(p)
            # Одна форма — разные слова («Лебедева»: его или она; «Александра»): чаще встречающаяся в книге.
            # Словарь может не знать уменьшительного («Кирюху» → «кирюх»): тогда форма, которая есть в книге.
            fallback = self.unknown_base(low)
            if fallback not in forms and all(self.capitalized[f] == 0 for f in forms) and self.capitalized[fallback] > 0:
                forms[fallback] = forms[max(forms, key=lambda f: self.capitalized[f])]
            key = max(forms, key=lambda f: (self.capitalized[f], f == low))
            chosen = forms[key][0]
            keys.append(key)
            nominative = nominative and any(p.tag.case == "nomn" for p in forms[key])
            gender = "f" if chosen.tag.gender == "femn" else "m" if chosen.tag.gender == "masc" else None
            info.append((gender, next(r for r in ("Name", "Patr", "Surn") if r in chosen.tag.grammemes)))
        return " ".join(keys), info, family, nominative

    def unknown_base(self, low: str) -> str:
        for ending in CASE_ENDINGS:
            if low.endswith(ending) and len(low) - len(ending) >= 3:
                stem = low[: -len(ending)]
                for base in (stem, stem + "о", stem + "а", stem + "я", stem + "ь"):
                    if base != low and self.capitalized[base] >= max(2, self.capitalized[low] // 4):
                        return base
        return low

    def run(self, book: Book, per_section: bool) -> dict:
        self.statistics(book.paragraphs)
        candidates: dict[tuple, Candidate] = {}

        def add(key, kind, section, text, start, end) -> Candidate:
            scope = section if per_section else 0
            cand = candidates.get((scope, key)) or candidates.setdefault((scope, key), Candidate(key, kind, scope))
            cand.count += 1
            cand.sections[section] += 1
            if len(cand.examples) < 3 and (not cand.examples or cand.count in (5, 40)):
                cand.examples.append(snippet(text, start, end))
            return cand

        for section, text in book.paragraphs:
            words = list(WORD.finditer(text))
            mentions: list[Mention] = []
            i = 0
            while i < len(words):
                m = words[i]
                low = m.group().lower()
                title = None
                if low in TITLES:
                    parsed = self.first(low)
                    title = parsed.normal_form
                    j = i + 1
                    if j < len(words) and text[m.end():words[j].start()] == " " and self.is_name(words[j].group(), False):
                        i = j  # титул перед именем: «генерал Иволгин»
                        m = words[i]
                    else:
                        inflected = parsed.inflect({"nomn", "sing"})
                        cand = add(inflected.word if inflected else title, "title", section, text, m.start(), m.end())
                        cand.forms[m.group()] += 1
                        if parsed.tag.gender in ("masc", "femn"):
                            cand.genders["Title:" + ("f" if parsed.tag.gender == "femn" else "m")] += 1
                        mentions.append(Mention(m.start(), m.end(), cand, parsed.tag.case == "nomn"))
                        i += 1
                        continue
                if not self.is_name(m.group(), sentence_start(text, m.start())):
                    i += 1
                    continue
                run = [m]
                while i + 1 < len(words) and text[run[-1].end():words[i + 1].start()] == " " and self.is_name(words[i + 1].group(), False):
                    i += 1
                    run.append(words[i])
                key, info, family, nominative = self.chain_key([x.group() for x in run])
                cand = add(("семья " + key) if family else key, "family" if family else "name", section, text, run[0].start(), run[-1].end())
                cand.forms[text[run[0].start():run[-1].end()]] += 1
                for gender, role in info:
                    if role:
                        cand.roles[role] += 1
                        if gender:
                            cand.genders[f"{role}:{gender}"] += 1
                if title:
                    cand.titles[title] += 1
                mentions.append(Mention(run[0].start(), run[-1].end(), cand, nominative))
                i += 1
            self.attribute_speakers(text, mentions)
            present = {id(x.cand): x.cand for x in mentions}
            for a in present.values():
                for b in present.values():
                    if a is not b:
                        a.together[b.key] += 1
        return candidates

    def attribute_speakers(self, text: str, mentions: list[Mention]) -> None:
        """«— Реплика, — сказал князь. — Ещё реплика»: ремарки — нечётные куски между тире."""
        if not text.startswith(("—", "–")):
            return
        offset = 1
        for index, part in enumerate(DASH.split(text[1:])):
            start = text.find(part, offset)
            if start < 0:
                return
            end = start + len(part)
            offset = end
            if index % 2 == 0:
                continue
            inside = [m for m in mentions if start <= m.start < end and m.nominative]
            if not inside:
                continue
            past = [p for p in (self.first(w.group()) for w in WORD.finditer(part)) if p.tag.POS == "VERB" and "past" in p.tag.grammemes]
            if not past:
                continue
            speaker = inside[0].cand
            speaker.speaker += 1
            if past[0].tag.gender in ("masc", "femn"):
                speaker.genders["verb:" + ("f" if past[0].tag.gender == "femn" else "m")] += 1


def snippet(text: str, start: int, end: int, width: int = 70) -> str:
    left, right = max(0, start - width), min(len(text), end + width)
    return ("…" if left else "") + text[left:start] + "[[" + text[start:end] + "]]" + text[end:right] + ("…" if right < len(text) else "")


def gender_source(c: Candidate) -> tuple[str, str | None]:
    """Род и откуда он: глагол в ремарке («сказала») надёжнее всего, затем отчество, обращение, имя, фамилия
    («Ганя», «Ганечка» по словарю женского рода, но «— сказал Ганя»)."""
    for source in ("verb", "Patr", "Title", "Name", "Surn"):
        m, f = c.genders.get(source + ":m", 0), c.genders.get(source + ":f", 0)
        if m + f < (2 if source == "verb" else 1):
            continue
        if m >= 2 * f:
            return "m", source
        if f >= 2 * m:
            return "f", source
    return "?", None


def gender_of(c: Candidate) -> str:
    return gender_source(c)[0]


def display(c: Candidate) -> str:
    return c.forms.most_common(1)[0][0] if c.forms else c.key


def detect_collection(book: Book, candidates: dict) -> bool:
    """Сборник: у разных разделов разные персонажи. В романе главные персонажи проходят через много разделов."""
    if len(book.sections) < 3:
        return False
    named = sorted((c for c in candidates.values() if c.kind == "name"), key=lambda c: -c.count)[:30]
    if not named:
        return False
    spread = sum(1 for c in named if len(c.sections) >= max(2, len(book.sections) // 4))
    return spread / len(named) < 0.3


# ---------------------------------------------------------------- запрос к LLM

PROMPT = """Ты помогаешь подготовить {what} к озвучке разными голосами.
Ниже — кандидаты, найденные в тексте автоматически: имена, фамилии, отчества, уменьшительные формы
и обращения («князь», «генеральша»). У каждого кандидата: номер, число упоминаний, сколько раз он
назван в ремарке диалога («— сказал князь»), род по тексту, обращения перед ним, с кем встречается
в одних абзацах, примеры.

Задача: собрать персонажей. Один персонаж часто записан по-разному: фамилия, имя и отчество,
уменьшительное имя, прозвище, обращение, разные падежи. Для каждого персонажа придумай постоянный
идентификатор латиницей (например "myshkin", "nastasya_filippovna") и перечисли ВСЕ номера кандидатов,
которые означают его.

Сопоставление должно быть железным: объединяй кандидатов, только если по тексту несомненно, что это
одно лицо (имя с отчеством стоят вместе, «Лиза» прямо названа «Елизаветой Петровной», примеры говорят
об одном человеке, одна и та же форма в разных падежах). Любое сомнение — кандидат в "other".
Лучше оставить настоящего персонажа в "other", чем склеить двух разных людей.

Правила:
- Каждый номер кандидата укажи ровно один раз: либо у одного персонажа, либо в "other".
- Первым в "candidates" ставь самого надёжного кандидата персонажа (самое частое имя).
- name — полное имя из найденных форм (например «Лев Николаевич Мышкин», если такие формы есть);
  ничего не додумывай: ни полных имён, которых нет в тексте, ни пояснений в скобках.
- В "other" — не персонажи (места, книги, исторические лица, которых только упоминают), семьи
  во множественном числе, и всё, что нельзя уверенно отнести к одному лицу.
- Обращение без имени («князь», «генерал») отнеси к персонажу, которого им называют почти всегда;
  если так называют нескольких — в "other".
- Отец и сын, муж и жена с одной фамилией — разные персонажи; общую фамилию без имени, если по
  примерам не ясно, кто это, отправь в "other".
- gender: "m", "f" или "?" — по тексту.
- Не придумывай номера, которых нет в списке. Ответ — только JSON без пояснений:
{{"characters": [{{"id": "...", "name": "полное имя", "gender": "m", "candidates": ["{p}1", "{p}7"]}}], "other": ["{p}9"]}}

Кандидаты:
{lines}
"""

VERIFY = """Проверка сопоставления персонажей в {what}.
Для каждого персонажа ниже: главный кандидат и остальные, которых к нему отнесли, с примерами из текста.
Для каждого остального кандидата ответь, тот ли это человек, что и главный:
"same" — несомненно тот же человек; "different" — другой человек или не человек; "unsure" — нельзя
уверенно сказать по примерам. Сомнение — это "unsure". Ответ — только JSON без пояснений:
{{"checks": [{{"candidate": "{p}5", "verdict": "same"}}]}}

{groups}
"""

SCHEMA = {
    "type": "object",
    "properties": {
        "characters": {"type": "array", "items": {"type": "object", "properties": {
            "id": {"type": "string"}, "name": {"type": "string"}, "gender": {"type": "string", "enum": ["m", "f", "?"]},
            "candidates": {"type": "array", "items": {"type": "string"}}},
            "required": ["id", "name", "gender", "candidates"]}},
        "other": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["characters", "other"],
}


def candidate_line(cid: str, c: Candidate, ids: dict) -> str:
    bits = [f"{cid} | {display(c)}", f"упоминаний {c.count}"]
    if c.speaker:
        bits.append(f"в ремарках {c.speaker}")
    bits.append(f"род {gender_of(c)}")
    if c.kind == "family":
        bits.append("мн. число (семья?)")
    if c.kind == "title":
        bits.append("обращение без имени")
    roles = [r for r, _ in c.roles.most_common()]
    if roles:
        bits.append("+".join({"Name": "имя", "Patr": "отчество", "Surn": "фамилия"}[r] for r in roles))
    if c.titles:
        bits.append("перед ним: " + ", ".join(t for t, _ in c.titles.most_common(3)))
    near = [ids[(c.scope, k)] for k, _ in c.together.most_common(8) if (c.scope, k) in ids][:4]
    if near:
        bits.append("рядом: " + ", ".join(near))
    line = " | ".join(bits)
    for e in c.examples[: 1 if c.count < 30 else 2]:
        line += f"\n    пример: {e}"
    return line


def extract(args) -> None:
    started = time.time()
    book = read_epub(args.book)
    extractor = Extractor()
    found = extractor.run(book, per_section=False)
    collection = args.scope == "section" or (args.scope == "auto" and detect_collection(book, found))
    if collection:
        found = Extractor().run(book, per_section=True)
    groups: dict[int, list] = collections.defaultdict(list)
    for c in found.values():
        if c.count >= args.min_count or c.speaker > 0:
            groups[c.scope].append(c)
    os.makedirs(args.out, exist_ok=True)
    ids, records, requests = {}, [], []
    for scope in sorted(groups):
        ranked = sorted(groups[scope], key=lambda c: (-(c.count + 3 * c.speaker), c.key))[: args.max_candidates]
        prefix = f"s{scope + 1}c" if collection else "c"
        for n, c in enumerate(ranked, 1):
            ids[(scope, c.key)] = f"{prefix}{n}"
        for c in ranked:
            records.append({
                "id": ids[(scope, c.key)], "scope": book.sections[scope]["id"] if collection else "book", "key": c.key,
                "display": display(c), "kind": c.kind, "count": c.count, "speaker": c.speaker, "gender": gender_of(c),
                "gender_source": gender_source(c)[1],
                "roles": dict(c.roles), "titles": dict(c.titles), "forms": dict(c.forms.most_common()),
                "sections": [book.sections[s]["id"] for s in sorted(c.sections)],
                "together": [ids[(scope, k)] for k, _ in c.together.most_common(8) if (scope, k) in ids],
                # Сколько абзацев с каждым кандидатом: голос делят только те, кто почти не встречается.
                "together_counts": {ids[(scope, k)]: n for k, n in c.together.most_common() if (scope, k) in ids},
                "examples": c.examples,
            })
        if not ranked:
            continue
        what = f"рассказ «{book.sections[scope]['title']}» из книги «{book.title}»" if collection else f"книгу «{book.title}»"
        prompt = PROMPT.format(what=what, p=prefix, lines="\n".join(candidate_line(ids[(scope, c.key)], c, ids) for c in ranked))
        name = book.sections[scope]["id"] if collection else "book"
        requests.append({"name": name, "title": book.sections[scope]["title"] if collection else book.title,
                         "sections": [name] if collection else [s["id"] for s in book.sections],
                         "candidates": [ids[(scope, c.key)] for c in ranked], "prompt": prompt})
        with open(os.path.join(args.out, f"llm_prompt_{name}.txt"), "w", encoding="utf-8") as f:
            f.write(prompt)
    # Отпечатки предложений → раздел. Повтор в разных разделах раздела не указывает — такие убираются.
    index: dict[str, str | None] = {}
    for section, text in book.paragraphs:
        sid = book.sections[section]["id"]
        for fp in sentence_fingerprints(text):
            index[fp] = sid if index.get(fp, sid) == sid else None
    with open(args.book, "rb") as f:
        file_sha = hashlib.sha256(f.read()).hexdigest()
    content_sha = hashlib.sha256("\n".join(letters(t) for _, t in book.paragraphs).encode()).hexdigest()
    identity = {"title": book.title, "author": book.author, "file_sha256": file_sha, "content_sha256": content_sha}
    save(args.out, "candidates.json", {"book": book.title, "scope": "section" if collection else "book",
                                       "sections": book.sections, "candidates": records})
    save(args.out, "llm_request.json", {"book": book.title, "scope": "section" if collection else "book",
                                        "response_schema": SCHEMA, "requests": requests})
    save(args.out, "book_index.json", {"book": identity, "sections": book.sections,
                                       "fingerprint": {"algorithm": "fnv1a64-48", "min_letters": 24},
                                       "sentences": {k: v for k, v in index.items() if v}})
    sizes = [len(r["prompt"]) for r in requests]
    print(f"«{book.title}»: разделов {len(book.sections)}, абзацев {len(book.paragraphs)}, "
          f"{'сборник — персонажи по разделам' if collection else 'роман — персонажи на всю книгу'}, "
          f"кандидатов {len(records)}, запросов {len(requests)} (до {max(sizes, default=0)} знаков), {time.time() - started:.1f} с")
    by_id = {r["id"]: r for r in records}
    for r in requests[: args.show_requests]:
        print(f"  [{r['name']}] {r['title']}: " + ", ".join(
            f"{by_id[c]['display']}({by_id[c]['count']},{by_id[c]['gender']})" for c in r["candidates"][: args.show]))


def save(folder: str, name: str, data) -> None:
    with open(os.path.join(folder, name), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)


# ---------------------------------------------------------------- LLM: Ollama Cloud, API DeepSeek или Groq

HERE = os.path.dirname(os.path.abspath(__file__))
USER_AGENT = "mytts-books/1.0 (+https://github.com/davnozdu/mytts-books)"
PROVIDERS = {
    "ollama": {"endpoint": "https://ollama.com", "model": "deepseek-v4.1-flash", "key": "OLLAMA_API_KEY"},
    "deepseek": {"endpoint": "https://api.deepseek.com", "model": "deepseek-flash", "key": "DEEPSEEK_API_KEY"},
    "groq": {"endpoint": "https://api.groq.com/openai/v1", "model": "qwen/qwen3.8-27b", "key": "GROQ_API_KEY"},
}
# Предел ответа по умолчанию: с размышлением / без. У Groq (Qwen3.8-27B) выход не больше 16384 токенов.
MAX_TOKENS = {
    "ollama": {"think": 80000, "plain": 16000},
    "deepseek": {"think": 64000, "plain": 16000},
    "groq": {"think": 16000, "plain": 16000},
}
# Запросов в минуту: у Groq free-тир всего 30/мин, 1000/день и 8K токенов/мин — темп режем, чтобы не упираться в лимит.
REQUESTS_PER_MINUTE = {"ollama": 120, "deepseek": 120, "groq": 10}


class LLMError(RuntimeError):
    """Ошибка сервиса LLM: код HTTP, текст ответа и (для 429) сколько секунд ждать до повтора."""

    def __init__(self, status: int, message: str, retry_after: str | None = None) -> None:
        super().__init__(f"HTTP {status}: {message}")
        self.status = status
        self.retry_after = retry_after

    def wait_seconds(self, default: float) -> float:
        """Сколько ждать до повтора: заголовок Retry-After сервиса (не больше двух минут), иначе заданное."""
        if not self.retry_after:
            return default
        try:
            return min(120.0, max(1.0, float(self.retry_after)))
        except ValueError:
            return default


class Pace:
    """Не чаще N запросов в минуту: потоки ждут очереди, а не бьют в лимит провайдера одновременно."""

    def __init__(self, per_minute: int) -> None:
        self.gap = 60.0 / per_minute if per_minute > 0 else 0.0
        self.lock = threading.Lock()
        self.next_at = 0.0

    def wait(self) -> None:
        if self.gap <= 0:
            return
        with self.lock:
            now = time.monotonic()
            delay = max(0.0, self.next_at - now)
            self.next_at = max(now, self.next_at) + self.gap
        if delay > 0:
            time.sleep(delay)


def load_env() -> None:
    """Ключи из файла .env рядом со скриптом или в текущей папке (строки KEY=VALUE); переменные окружения главнее."""
    for folder in (HERE, os.getcwd()):
        path = os.path.join(folder, ".env")
        if not os.path.isfile(path):
            continue
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                name, value = line.split("=", 1)
                os.environ.setdefault(name.strip(), value.strip().strip("\"'"))


def request_chat(provider: str, endpoint: str, model: str, key: str, prompt: str, think: bool, max_tokens: int, timeout: int) -> dict:
    """Один запрос. Возвращает content, thinking (длина), причину остановки и токены в общем виде."""
    base = endpoint.rstrip("/")
    if provider == "ollama":
        url = base + "/api/chat"
        body = {"model": model, "stream": False, "think": think, "options": {"temperature": 0, "num_predict": max_tokens},
                "messages": [{"role": "user", "content": prompt}]}
        headers = {"Content-Type": "application/json", "Authorization": "Bearer " + key}
    else:
        url = base + "/chat/completions"
        body = {"model": model, "stream": False, "max_tokens": max_tokens, "temperature": 0,
                "messages": [{"role": "user", "content": prompt}]}
        if provider == "deepseek":
            # Оригинальный API DeepSeek (api.deepseek.com): размышление — thinking, ответ — в reasoning_content.
            body["thinking"] = {"type": "enabled" if think else "disabled"}
            if think:
                body["reasoning_effort"] = "high"
        elif provider == "groq":
            # Groq (Qwen): размышление — это reasoning_effort; "none" выключает, ответ — в message.reasoning.
            # User-Agent обязателен: без него Cloudflare отдаёт 403 (error code 1010), а не 401.
            body["reasoning_effort"] = "medium" if think else "none"
        headers = {"Content-Type": "application/json", "Authorization": "Bearer " + key, "User-Agent": USER_AGENT}
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            data = json.loads(response.read())
    except urllib.error.HTTPError as e:  # текст ошибки сервиса, без ключа
        retry = e.headers.get("retry-after") if e.headers else None
        raise LLMError(e.code, e.read().decode("utf-8", "replace")[:300], retry) from None
    if provider == "ollama":
        message = data.get("message", {})
        return {"content": message.get("content", ""), "thinking_chars": len(message.get("thinking") or ""),
                "done_reason": data.get("done_reason"), "prompt_tokens": data.get("prompt_eval_count"),
                "output_tokens": data.get("eval_count")}
    choice = data["choices"][0]
    message = choice.get("message", {})
    usage = data.get("usage", {})
    # DeepSeek кладёт размышление в reasoning_content, Groq (Qwen) — в reasoning; берём то, что есть.
    thinking = message.get("reasoning_content") or message.get("reasoning") or ""
    return {"content": message.get("content") or "", "thinking_chars": len(thinking),
            "done_reason": choice.get("finish_reason"), "prompt_tokens": usage.get("prompt_tokens"),
            "output_tokens": usage.get("completion_tokens")}


def llm(args) -> None:
    load_env()
    preset = PROVIDERS[args.provider]
    args.model = args.model or preset["model"]
    args.endpoint = args.endpoint or preset["endpoint"]
    key = os.environ.get(preset["key"], "")
    if not key:
        sys.exit(f"Нет ключа: задайте {preset['key']} в переменной окружения или в файле .env (см. README.md)")
    request = json.load(open(os.path.join(args.dir, "llm_request.json"), encoding="utf-8"))
    by_id = {c["id"]: c for c in json.load(open(os.path.join(args.dir, "candidates.json"), encoding="utf-8"))["candidates"]}
    folder = os.path.join(args.dir, "answers")
    os.makedirs(folder, exist_ok=True)
    pace = Pace(REQUESTS_PER_MINUTE.get(args.provider, 60))

    def chat(prompt: str, name: str) -> str:
        """Один запрос; ответ в answers/NAME.json, служебные поля (причина остановки, токены) — в NAME.meta.json.
        Размышление по умолчанию включено (--no-think выключает): точнее, роман ~3 мин вместо секунд.
        Лимит провайдера (429) — ждём Retry-After и повторяем; пустой ответ (токены ушли на размышление) — тоже."""
        limit = args.max_tokens or MAX_TOKENS.get(args.provider, MAX_TOKENS["ollama"])["think" if args.think else "plain"]
        delay = 2.0
        for attempt in range(1, 4):
            pace.wait()
            started = time.time()
            try:
                reply = request_chat(args.provider, args.endpoint, args.model, key, prompt, args.think, limit, args.timeout)
            except LLMError as e:
                if attempt >= 3 or e.status not in (429, 500, 502, 503, 504):
                    raise
                delay = e.wait_seconds(delay * 2)
                print(f"  [{name}] {e}; ждём {delay:.0f} с и повторяем ({attempt + 1}/3)", flush=True)
                time.sleep(delay)
                continue
            content = reply.pop("content")
            meta = dict(reply, provider=args.provider, model=args.model, content_chars=len(content),
                        seconds=round(time.time() - started, 1), attempt=attempt)
            save(folder, name + ".meta.json", meta)
            with open(os.path.join(folder, name + ".json"), "w", encoding="utf-8") as f:
                f.write(content)
            if content.strip() or attempt >= 3:
                break
        return (f"{meta['seconds']} с, ответ {meta['content_chars']} знаков, размышление {meta['thinking_chars']}, "
                f"токенов {meta['prompt_tokens']}→{meta['output_tokens']}, {meta['done_reason']}")

    def ask(r: dict) -> str:
        line = chat(r["prompt"], r["name"])
        answer = load_answer(os.path.join(folder, r["name"] + ".json"))
        prompt = verify_prompt(r, answer, by_id, request["scope"] == "section") if answer else None
        if not prompt:
            return line + "; проверять нечего"
        return line + "; проверка: " + chat(prompt, r["name"] + ".verify")

    started = time.time()
    todo = [r for r in request["requests"] if args.redo or load_answer(os.path.join(folder, r["name"] + ".json")) is None]
    pace_note = f", темп ≤{REQUESTS_PER_MINUTE.get(args.provider, 60)}/мин" if REQUESTS_PER_MINUTE.get(args.provider, 60) < 60 else ""
    print(f"Запросов {len(todo)} из {len(request['requests'])} (остальные уже есть), {args.provider}: {args.model}, "
          f"размышление {'вкл' if args.think else 'выкл'}, одновременно {args.parallel}{pace_note}", flush=True)
    with concurrent.futures.ThreadPoolExecutor(args.parallel) as pool:
        for r, line in zip(todo, pool.map(lambda r: safe(ask, r), todo)):
            print(f"  [{r['name']}] {r['title']}: {line}", flush=True)
    print(f"Готово за {time.time() - started:.1f} с")


def safe(fn, r) -> str:
    try:
        return fn(r)
    except Exception as e:  # сетевой сбой одного рассказа не останавливает остальные
        return f"ошибка: {type(e).__name__}: {str(e)[:160]}"


def verify_prompt(r: dict, answer: dict, by_id: dict, collection: bool) -> str | None:
    """Второй запрос: каждую склейку подтверждает отдельный ответ «тот же / другой / не уверена»."""
    groups = []
    for raw in answer.get("characters", []):
        refs = [ref for ref in raw.get("candidates", []) if ref in by_id and ref in r["candidates"]]
        if len(refs) < 2:
            continue
        lines = [f"Персонаж «{raw.get('name', '')}». Главный: {describe(by_id[refs[0]])}"]
        lines += [f"  проверить {describe(by_id[ref])}" for ref in refs[1:]]
        groups.append("\n".join(lines))
    if not groups:
        return None
    what = f"рассказе «{r['title']}»" if collection else f"книге «{r['title']}»"
    prefix = r["candidates"][0].rstrip("0123456789")
    return VERIFY.format(what=what, p=prefix, groups="\n\n".join(groups))


def describe(c: dict) -> str:
    text = f"{c['id']} {c['display']} (упоминаний {c['count']}, род {c['gender']}, формы: {', '.join(list(c['forms'])[:4])})"
    for e in c["examples"][:2]:
        text += f"\n      пример: {e}"
    return text


# ---------------------------------------------------------------- проверка ответа

def load_answer(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    text = open(path, encoding="utf-8").read()
    start, end = text.find("{"), text.rfind("}")
    try:
        return json.loads(text[start:end + 1]) if 0 <= start < end else None
    except json.JSONDecodeError:
        return None


def build_cast(r: dict, answer: dict | None, verdicts: dict | None, candidates: dict, prefix: str) -> dict:
    """Проверенный ответ: каждый кандидат ровно у одного персонажа или в «прочих». К персонажу остаются
    только подтверждённые второй проверкой («same») и не противоречащие ему по роду; остальное — «прочие»."""
    problems: list[str] = []
    dropped: list[str] = []
    own_ids = set(r["candidates"])
    used: dict[str, str] = {}
    characters = []
    if answer is None:
        problems.append("нет ответа LLM или в нём нет JSON — все кандидаты в «прочих»")
        answer = {}
    for raw in answer.get("characters", []):
        cid = re.sub(r"[^a-z0-9_]", "_", str(raw.get("id", "")).lower()).strip("_")
        cid = prefix + cid if cid else ""
        if not cid or cid.endswith(("author", "other")) or any(ch["id"] == cid for ch in characters):
            problems.append(f"идентификатор «{raw.get('id')}» пустой, служебный или повторяется — персонаж пропущен")
            continue
        own = []
        gender = raw.get("gender") if raw.get("gender") in ("m", "f", "?") else "?"
        for ref in raw.get("candidates", []):
            if ref not in own_ids:
                problems.append(f"{cid}: несуществующий кандидат {ref}")
            elif ref in used:
                problems.append(f"{ref} указан дважды ({used[ref]} и {cid}) — оставлен у {used[ref]}")
            elif own and verdicts is not None and verdicts.get(ref) != "same":
                dropped.append(f"{ref} {candidates[ref]['display']} → прочие: проверка «{verdicts.get(ref, 'нет ответа')}» для {cid}")
            elif own and gender in ("m", "f") and candidates[ref]["gender"] in ("m", "f") and candidates[ref]["gender"] != gender \
                    and candidates[ref].get("gender_source") in ("verb", "Patr", "Title"):
                # Только надёжный род: глагол в ремарке, отчество, обращение. Уменьшительные («Ганечка») словарь
                # часто считает женскими — по имени род не решает.
                dropped.append(f"{ref} {candidates[ref]['display']} → прочие: род {candidates[ref]['gender']}, у {cid} {gender}")
            else:
                used[ref] = cid
                own.append(ref)
        if not own:
            problems.append(f"{cid}: нет ни одного кандидата — пропущен")
            continue
        characters.append({"id": cid, "name": str(raw.get("name") or ""), "gender": gender, "candidates": own})
    # Обращение без имени («генерал», «господин») — только тому, перед чьим именем оно стоит почти всегда.
    for ch in characters:
        for ref in [ref for ref in ch["candidates"][1:] if candidates[ref]["kind"] == "title"]:
            word = candidates[ref]["key"]
            per = {other["id"]: sum(candidates[x]["titles"].get(word, 0) for x in other["candidates"] if x != ref) for other in characters}
            total = sum(per.values())
            if total == 0 or per[ch["id"]] < 0.8 * total:
                ch["candidates"].remove(ref)
                used[ref] = "other"
                other_owner = max(per, key=per.get) if total else None
                dropped.append(f"{ref} {candidates[ref]['display']} → прочие: обращение стоит перед именем {ch['id']} "
                               f"{per[ch['id']]} из {total} раз" + (f" (чаще {other_owner})" if other_owner and other_owner != ch["id"] else ""))
    for ch in characters:
        # Имя только из слов книги: «Аглая Ивановна Епанчина» (все слова найдены) — да; «Анастасия (Настенька)»,
        # если «Анастасии» в книге нет, — нет. Тогда самая полная найденная форма в именительном падеже.
        words = {w for ref in ch["candidates"] for w in candidates[ref]["key"].split()}
        name_words = [w.lower().replace("ё", "е") for w in re.findall(r"[А-ЯЁа-яё-]+", ch["name"])]
        if not name_words or any(w not in words for w in name_words) or re.search(r"[()]", ch["name"]):
            longest = max((candidates[ref] for ref in ch["candidates"] if candidates[ref]["kind"] == "name"),
                          key=lambda c: (len(c["key"].split()), c["count"]), default=candidates[ch["candidates"][0]])
            ch["name"] = " ".join(w.capitalize() if longest["kind"] == "name" else w for w in longest["key"].split())
    other = []
    for ref in answer.get("other", []):
        if ref in own_ids and ref not in used:
            used[ref] = "other"
            other.append(ref)
    for line in dropped:
        ref = line.split()[0]
        if used.get(ref, "other") == "other" and ref not in other:
            used[ref] = "other"
            other.append(ref)
    missing = [ref for ref in r["candidates"] if ref not in used]
    if missing:
        problems.append(f"не распределено {len(missing)} — отнесены к «прочим»: {', '.join(missing[:15])}")
        other += missing
    alias: dict[str, set] = collections.defaultdict(set)
    for ch in characters:
        forms = collections.Counter()
        ch["mentions"] = sum(candidates[ref]["count"] for ref in ch["candidates"])
        ch["speaker"] = sum(candidates[ref]["speaker"] for ref in ch["candidates"])
        for ref in ch["candidates"]:
            forms.update(candidates[ref]["forms"])
            alias[candidates[ref]["key"]].add(ch["id"])
            for form in candidates[ref]["forms"]:
                alias[form.lower().replace("ё", "е")].add(ch["id"])
        ch["forms"] = [f for f, _ in forms.most_common()]
    for ref in other:
        alias[candidates[ref]["key"]].add("other")
    characters.sort(key=lambda ch: (-(ch["speaker"] * 3 + ch["mentions"]), ch["id"]))
    return {
        "sections": r["sections"], "title": r["title"], "characters": characters,
        "other": [{"candidate": ref, "display": candidates[ref]["display"], "count": candidates[ref]["count"]} for ref in other],
        # Форма → персонаж. Если форма у нескольких (отец и сын «Иволгин»), читалка решает по контексту,
        # а без него читает голосом «прочих».
        "alias_index": {k: next(iter(v)) for k, v in sorted(alias.items()) if len(v) == 1},
        "ambiguous": {k: sorted(v) for k, v in sorted(alias.items()) if len(v) > 1},
        "problems": problems,
        "dropped": dropped,
    }


def apply(args) -> None:
    data = json.load(open(os.path.join(args.dir, "candidates.json"), encoding="utf-8"))
    request = json.load(open(os.path.join(args.dir, "llm_request.json"), encoding="utf-8"))
    candidates = {c["id"]: c for c in data["candidates"]}
    collection = request["scope"] == "section"
    casts = []
    for r in request["requests"]:
        answer = load_answer(os.path.join(args.dir, "answers", r["name"] + ".json"))
        checked = load_answer(os.path.join(args.dir, "answers", r["name"] + ".verify.json"))
        verdicts = {c.get("candidate"): c.get("verdict") for c in checked.get("checks", [])} if checked else None
        if verdicts is None and answer and any(len(ch.get("candidates", [])) > 1 for ch in answer.get("characters", [])):
            verdicts = {}  # склейки без второй проверки не принимаются
        casts.append(build_cast(r, answer, verdicts, candidates, r["name"] + "." if collection else ""))
    section_cast = {sid: i for i, cast in enumerate(casts) for sid in cast["sections"]}
    save(args.dir, "cast.json", {
        "book": data["book"], "scope": request["scope"], "narrator": "author", "others": "other",
        "sections": [dict(s, cast=section_cast.get(s["id"])) for s in data["sections"]], "casts": casts})
    total = sum(len(c["characters"]) for c in casts)
    print(f"«{data['book']}»: наборов {len(casts)}, персонажей {total}, "
          f"в «прочих» {sum(len(c['other']) for c in casts)}, снято проверкой {sum(len(c['dropped']) for c in casts)}, "
          f"замечаний {sum(len(c['problems']) for c in casts)}")
    for cast in casts[: args.show_casts]:
        print(f"  [{','.join(cast['sections'][:3])}{'…' if len(cast['sections']) > 3 else ''}] {cast['title']}")
        for ch in cast["characters"][: args.show]:
            print(f"     {ch['id']:<28} {ch['name']:<32} {ch['gender']} упоминаний={ch['mentions']:<4} реплик={ch['speaker']:<3} "
                  f"{', '.join(ch['forms'][:5])}")
        for p in cast["dropped"]:
            print("     - " + p)
        for p in cast["problems"]:
            print("     ! " + p)


# ---------------------------------------------------------------- голоса

def voices(args) -> None:
    """Свой голос — главным: тем, кто говорит (реплики в ремарках) или часто упоминается; по полу, по одному
    на персонажа. Когда голоса нужного пола кончились — общий голос с тем, с кем персонаж почти не встречается
    в одних абзацах. Остальные (второстепенные, молчащие, без пола) — голос «прочих» своего пола или автора."""
    cast = json.load(open(os.path.join(args.dir, "cast.json"), encoding="utf-8"))
    candidates = {c["id"]: c for c in json.load(open(os.path.join(args.dir, "candidates.json"), encoding="utf-8"))["candidates"]}
    spec = json.load(open(args.voices, encoding="utf-8"))
    reserved = {spec["narrator"], spec["other_m"], spec["other_f"]}
    pool = {g: [v["id"] for v in spec["voices"] if v["gender"] == g and v["id"] not in reserved] for g in ("m", "f")}
    unknown = [v["id"] for v in spec["voices"] if v["gender"] not in ("m", "f")]

    def meetings(a: dict, b: dict) -> int:
        return sum(candidates[x].get("together_counts", {}).get(y, 0) for x in a["candidates"] for y in b["candidates"])

    for group in cast["casts"]:
        order = sorted(group["characters"], key=lambda ch: (-ch["speaker"], -ch["mentions"], ch["id"]))
        holders: dict[str, list] = collections.defaultdict(list)
        for ch in order:
            main = ch["gender"] in ("m", "f") and (ch["speaker"] >= args.min_speaker or ch["mentions"] >= args.min_mentions)
            if not main:
                ch["voice"] = spec["other_" + ch["gender"]] if ch["gender"] in ("m", "f") else spec["narrator"]
                ch["role"] = "other"
                continue
            free = [v for v in pool[ch["gender"]] if not holders[v]]
            if free:
                voice, role = free[0], "own"
            else:
                # Голос того, с кем меньше всего общих абзацев; слишком часто вместе — в «прочие».
                voice = min(pool[ch["gender"]], key=lambda v: max(meetings(ch, other) for other in holders[v]))
                shared = max(meetings(ch, other) for other in holders[voice])
                role = "shared" if shared <= args.max_shared else "other"
                if role == "other":
                    voice = spec["other_" + ch["gender"]]
            ch["voice"], ch["role"] = voice, role
            if role != "other":
                holders[voice].append(ch)
        group["voices"] = {"narrator": spec["narrator"], "other_m": spec["other_m"], "other_f": spec["other_f"]}
    cast["voice_model"] = spec.get("model", "")
    save(args.dir, "cast.json", cast)
    roles = collections.Counter(ch["role"] for g in cast["casts"] for ch in g["characters"])
    print(f"«{cast['book']}»: свой голос {roles['own']}, общий {roles['shared']}, «прочие» {roles['other']}; "
          f"голосов: мужских {len(pool['m'])}, женских {len(pool['f'])} (+ автор {spec['narrator']}, прочие "
          f"{spec['other_m']}/{spec['other_f']}), без пола не раздаются: {', '.join(unknown) or 'нет'}")
    for group in cast["casts"][: args.show_casts]:
        print(f"  [{','.join(group['sections'][:3])}{'…' if len(group['sections']) > 3 else ''}] {group['title']}")
        for ch in sorted(group["characters"], key=lambda ch: (ch["role"] == "other", -ch["speaker"], -ch["mentions"]))[: args.show]:
            print(f"     {ch['voice']:<14} {ch['role']:<6} {ch['name']:<32} {ch['gender']} реплик={ch['speaker']:<3} упоминаний={ch['mentions']}")


# ---------------------------------------------------------------- обмен

def export(args) -> None:
    """cast.json + отпечатки → .mytts-book: без текста книги и примеров, можно передавать другим."""
    cast = json.load(open(os.path.join(args.dir, "cast.json"), encoding="utf-8"))
    index = json.load(open(os.path.join(args.dir, "book_index.json"), encoding="utf-8"))
    if "sentences" not in index:
        sys.exit("book_index.json старого формата: повторите extract")
    by_section: dict[str, list] = collections.defaultdict(list)
    for fp, sid in sorted(index["sentences"].items()):
        by_section[sid].append(fp)
    casts = []
    for group in cast["casts"]:
        characters = []
        for ch in group["characters"]:
            if ch.get("role") == "other" and not args.keep_other:
                continue
            item = {"id": ch["id"], "name": ch["name"], "gender": ch["gender"], "speaker": ch["speaker"],
                    "mentions": ch["mentions"], "forms": ch["forms"][: args.max_forms]}
            if ch.get("voice") and ch.get("role") in ("own", "shared"):
                item["voice_hint"] = ch["voice"]
            characters.append(item)
        other = [o["display"] for o in group["other"]] + [ch["name"] for ch in group["characters"]
                                                          if ch.get("role") == "other" and not args.keep_other]
        casts.append({"sections": group["sections"], "characters": characters, "other": other})
    data = {
        "format": "mytts-book", "version": 1, "book": index["book"], "scope": cast["scope"],
        "voice_model": cast.get("voice_model", ""),
        "sections": [{"id": s["id"], "title": s["title"], "cast": s.get("cast")} for s in cast["sections"]],
        "casts": casts, "fingerprint": index["fingerprint"], "fingerprints": dict(by_section),
    }
    target = args.output or os.path.join(args.dir, re.sub(r"[^\w.-]+", "_", index["book"]["title"]) + ".mytts-book")
    with open(target, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
    total = sum(len(v) for v in by_section.values())
    print(f"{target}: персонажей {sum(len(c['characters']) for c in casts)}, разделов {len(data['sections'])}, "
          f"отпечатков {total}, {os.path.getsize(target) / 1024:.0f} КБ")


def vectors(args) -> None:
    """Тест-векторы отпечатков для Kotlin (books/BookFingerprint.kt)."""
    samples = [
        "— Да, князь, — сказал Рогожин. Он помолчал и прибавил: «Ёлки-палки, вот так встреча!»",
        "Князь Лев Николаевич Мы́шкин вошёл в гостиную; Настасья Филипповна обернулась к нему.",
        "Коротко. Совсем коротко! Но вот это предложение уже достаточно длинное для отпечатка?..",
        "В 1867 году в Петербурге было сыро и мокро… Поезд подходил к Варшавскому вокзалу.",
        "Mixed text with Latin words, digits 42 and русские слова\u00a0вместе — проверка нормализации.",
        "Неразрывный пробел после точки тоже граница.\u00a0Второе предложение начинается сразу после него!",
    ]
    data = [{"text": t, "letters": [letters(x)[:48] for x in SENTENCES.split(t)], "fingerprints": sentence_fingerprints(t)} for t in samples]
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    print(f"{args.output}: {sum(len(d['fingerprints']) for d in data)} отпечатков")


# ---------------------------------------------------------------- всё сразу

def process(args) -> None:
    """EPUB → .mytts-book: extract, llm, apply, voices, export с настройками по умолчанию."""
    started = time.time()
    stem = re.sub(r"[^\w.-]+", "_", os.path.splitext(os.path.basename(args.book))[0])[:80]
    out = args.out or os.path.join("out", stem)
    step = lambda title: print(f"\n== {title}", flush=True)
    step("1/5 Кандидаты в персонажи")
    extract(Namespace(book=args.book, out=out, scope=args.scope, min_count=2, max_candidates=200, show=8, show_requests=5))
    step(f"2/5 Сопоставление в LLM ({args.provider})")
    llm(Namespace(dir=out, provider=args.provider, model=args.model, endpoint=args.endpoint, parallel=args.parallel,
                  redo=args.redo, timeout=args.timeout, max_tokens=args.max_tokens, think=args.think))
    step("3/5 Проверка ответа")
    apply(Namespace(dir=out, show=12, show_casts=3))
    step("4/5 Голоса")
    voices(Namespace(dir=out, voices=args.voices, min_speaker=2, min_mentions=30, max_shared=2, show=12, show_casts=3))
    step("5/5 Файл для MyTTS")
    export(Namespace(dir=out, output=args.output, max_forms=24, keep_other=False))
    print(f"\nГотово за {time.time() - started:.0f} с. Перенесите файл .mytts-book на телефон и загрузите его в MyTTS: "
          f"настройки LLM → мультиголос → «Книги с голосами персонажей» → «Загрузить файл книги».")


def add_llm_options(parser) -> None:
    parser.add_argument("--provider", choices=sorted(PROVIDERS), default="ollama",
                        help="ollama (Ollama Cloud), deepseek (API DeepSeek) или groq (Qwen 3.8 27B)")
    parser.add_argument("--model", help="по умолчанию: ollama — deepseek-v4.1-flash, deepseek — deepseek-flash, groq — qwen/qwen3.8-27b")
    parser.add_argument("--endpoint", help="адрес сервиса, если не стандартный (например, свой сервер Ollama)")
    parser.add_argument("--parallel", type=int, default=1, help="одновременных запросов (по умолчанию по одному)")
    parser.add_argument("--redo", action="store_true", help="спросить заново и те разделы, на которые ответ уже есть")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--max-tokens", type=int, help="предел ответа: 16000, с --think 80000 (размышление входит в предел)")
    # Размышление по умолчанию включено: склейки персонажей заметно полнее и без ошибок (проверено на «Идиоте»).
    parser.add_argument("--no-think", dest="think", action="store_false", help="без размышления: в 5–15 раз быстрее, но менее точно")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    pr = sub.add_parser("process", help="всё сразу: EPUB → файл .mytts-book для импорта в MyTTS")
    pr.add_argument("book", help="файл книги .epub")
    pr.add_argument("-o", "--out", help="рабочая папка (по умолчанию out/<имя книги>)")
    pr.add_argument("--output", help="путь итогового .mytts-book (по умолчанию в рабочей папке)")
    pr.add_argument("--scope", choices=["auto", "book", "section"], default="auto", help="роман, сборник или определить")
    pr.add_argument("--voices", default=os.path.join(HERE, "voices", "silero_cis.json"), help="голоса модели с полом")
    add_llm_options(pr)
    pr.set_defaults(func=process)
    e = sub.add_parser("extract", help="EPUB → кандидаты, запросы к LLM, указатель абзацев")
    e.add_argument("book")
    e.add_argument("-o", "--out", required=True)
    e.add_argument("--scope", choices=["auto", "book", "section"], default="auto",
                   help="book — роман, section — сборник рассказов, auto — определить")
    e.add_argument("--min-count", type=int, default=2, help="минимум упоминаний (кроме названных в ремарках)")
    e.add_argument("--max-candidates", type=int, default=200, help="на один запрос")
    e.add_argument("--show", type=int, default=12)
    e.add_argument("--show-requests", type=int, default=30)
    e.set_defaults(func=extract)
    q = sub.add_parser("llm", help="отправить запросы в LLM (Ollama Cloud или DeepSeek)")
    q.add_argument("dir")
    add_llm_options(q)
    q.set_defaults(func=llm)
    a = sub.add_parser("apply", help="ответы LLM → cast.json")
    a.add_argument("dir")
    a.add_argument("--show", type=int, default=25)
    a.add_argument("--show-casts", type=int, default=30)
    a.set_defaults(func=apply)
    v = sub.add_parser("voices", help="раздать голоса персонажам cast.json")
    v.add_argument("dir")
    v.add_argument("--voices", default=os.path.join(HERE, "voices", "silero_cis.json"))
    v.add_argument("--min-speaker", type=int, default=2, help="свой голос: не меньше реплик в ремарках…")
    v.add_argument("--min-mentions", type=int, default=30, help="…или не меньше упоминаний (главный герой без ремарок)")
    v.add_argument("--max-shared", type=int, default=2, help="общий голос: не больше общих абзацев")
    v.add_argument("--show", type=int, default=40)
    v.add_argument("--show-casts", type=int, default=6)
    v.set_defaults(func=voices)
    x = sub.add_parser("export", help="cast.json + отпечатки → файл .mytts-book для MyTTS и обмена")
    x.add_argument("dir")
    x.add_argument("-o", "--output")
    x.add_argument("--max-forms", type=int, default=24)
    x.add_argument("--keep-other", action="store_true", help="оставить персонажей с ролью «прочие» в списке")
    x.set_defaults(func=export)
    t = sub.add_parser("vectors", help="тест-векторы отпечатков для приложения")
    t.add_argument("-o", "--output", default="fingerprint_vectors.json")
    t.set_defaults(func=vectors)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
