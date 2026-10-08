#!/usr/bin/env python3
"""MyTTS · подготовка книги к озвучке голосами персонажей.

Одна команда — EPUB на входе, файл .mytts-book на выходе (импорт в MyTTS: настройки LLM → мультиголос →
«Книги с голосами персонажей» → «Загрузить файл книги»):

  python mytts_book.py process book.epub                      # Ollama Cloud, ключ OLLAMA_API_KEY
  python mytts_book.py process book.epub --provider deepseek  # API DeepSeek, ключ DEEPSEEK_API_KEY

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
        self.normal_inside: collections.Counter = collections.Counter()

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
                        self.normal_inside[self.first(w).normal_form.replace("ё", "е")] += 1

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
            return self.capital_inside[k] > 0 or (tagged and self.lower[k] == 0) or (
                self.normal_inside[self.first(word).normal_form.replace("ё", "е")] >= 2 and self.lower[k] == 0)
        return tagged or (self.capital_inside[k] >= 1 and self.lower[k] == 0)

    POSSESSIVE = re.compile(r"^(.{2,}?)[иы]н(?:а|о|ы|ой|ою|ому|ым|ом|ых|ыми|у|е)?$")

    def possessive(self, k: str) -> bool:
        """«Зинина», «Анину», «Ксенин»: притяжательное от имени, которое в книге встречается намного чаще.
        Настоящая фамилия («Рогожин») остаётся: «Рогожа» в книге нет."""
        m = self.POSSESSIVE.match(k)
        if not m:
            return False
        owner = max(self.capitalized[m.group(1) + "а"], self.capitalized[m.group(1) + "я"])
        return owner >= max(5, 3 * self.capitalized[k])

    def chain_key(self, words: list[str], expected_gender: str | None = None) -> tuple[str, list, bool, bool]:
        """Ключ цепочки «Евгения Павловича» → «евгений павлович»: падеж и род согласуются по всем словам.
        Возвращает (ключ, [(род, роль)], семья, именительный падеж)."""
        options = [[p for p in self.named(w) if "sing" in p.tag.grammemes or "Sgtm" in p.tag.grammemes] for w in words]
        if expected_gender:
            options = [[p for p in opt if p.tag.gender == expected_gender] or opt for opt in options]
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
                    adjective = self.first(w)
                    base = self.unknown_base(low)
                    if adjective.tag.POS == "ADJF" and adjective.normal_form.endswith(("ский", "цкий", "ской", "цкой")):
                        base = adjective.normal_form.replace("ё", "е")
                    keys.append(base)
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
            if fallback != low and fallback not in forms and all(self.capitalized[f] == 0 for f in forms) and self.capitalized[fallback] > 0:
                forms[fallback] = forms[max(forms, key=lambda f: self.capitalized[f])]
            key = max(forms, key=lambda f: (self.capitalized[f], f == low))
            chosen = forms[key][0]
            keys.append(key)
            nominative = nominative and any(p.tag.case == "nomn" for p in forms[key])
            gender = "f" if chosen.tag.gender == "femn" else "m" if chosen.tag.gender == "masc" else None
            info.append((gender, next(r for r in ("Name", "Patr", "Surn") if r in chosen.tag.grammemes)))
        return " ".join(keys), info, family, nominative

    def unknown_base(self, low: str) -> str:
        # Adjectival surnames absent from the dictionary: «Тоцким» → «Тоцкий»,
        # only when that nominative form actually occurs in the same text.
        for ending in ("ого", "ому", "ыми", "ых", "им", "ым", "ом"):
            if low.endswith(ending):
                stem = low[:-len(ending)]
                for suffix in ("ий", "ый", "ой"):
                    base = stem + suffix
                    if len(stem) >= 3 and self.capitalized[base] >= max(2, self.capitalized[low] // 4):
                        return base
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
            if len(cand.examples) < 6 and (cand.count <= 3 or cand.count in (10, 40, 100)):
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
                parsed = self.first(low)
                if parsed.normal_form in TITLES:
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
                key, info, family, nominative = self.chain_key([x.group() for x in run],
                    parsed.tag.gender if title else None)
                cand = add(("семья " + key) if family else key, "family" if family else "name", section, text, run[0].start(), run[-1].end())
                cand.forms[text[run[0].start():run[-1].end()]] += 1
                for gender, role in info:
                    if role:
                        cand.roles[role] += 1
                        if gender:
                            cand.genders[f"{role}:{gender}"] += 1
                if title:
                    cand.titles[title] += 1
                    if parsed.tag.gender in ("masc", "femn"):
                        cand.genders["Title:" + ("f" if parsed.tag.gender == "femn" else "m")] += 1
                mentions.append(Mention(run[0].start(), run[-1].end(), cand, nominative))
                i += 1
            # Apposition also identifies titles: «Иван Петрович, отставной генерал».
            # Otherwise a title belonging to two people can look unique from prefixes alone.
            for mention in mentions:
                if mention.cand.kind != "title":
                    continue
                previous = [m for m in mentions if m.cand.kind == "name" and m.end < mention.start]
                if not previous:
                    continue
                name = previous[-1]
                gap = text[name.end:mention.start]
                if len(gap) <= 64 and re.fullmatch(r"\s*,\s*(?:[А-ЯЁа-яё-]+\s+){0,3}", gap):
                    qualifiers = [self.first(w.group()) for w in WORD.finditer(gap)]
                    if all(q.tag.POS in ("ADJF", "PRTF", "ADVB") for q in qualifiers):
                        name.cand.titles[mention.cand.key] += 1
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
            # A dash inside a spoken sentence is not necessarily a narrator's remark.
            # Only a nearby subject and past-tense verb in the opening clause count.
            clause = re.split(r"[.!?…;]", part, maxsplit=1)[0][:180]
            inside = [m for m in mentions if start <= m.start < start + len(clause)
                      and m.end <= start + len(clause) and m.nominative and m.cand.kind != "family"]
            if not inside:
                continue
            pairs = []
            for w in WORD.finditer(clause):
                parsed = self.first(w.group())
                if parsed.tag.POS != "VERB" or "past" not in parsed.tag.grammemes:
                    continue
                for mention in inside:
                    left, right = sorted(((mention.start-start, mention.end-start), (w.start(), w.end())))
                    gap = clause[left[1]:right[0]]
                    if len(gap) > 48 or re.search(r"[,.:;!?…]", gap):
                        continue
                    if any(other != mention and left[1] <= other.start-start < right[0] for other in inside):
                        continue
                    pairs.append((len(gap), mention, parsed))
            if not pairs:
                continue
            _, mention, verb = min(pairs, key=lambda pair: pair[0])
            speaker = mention.cand
            speaker.speaker += 1
            if verb.tag.gender in ("masc", "femn"):
                speaker.genders["verb:" + ("f" if verb.tag.gender == "femn" else "m")] += 1


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
{{"checks": [{{"character": "id персонажа", "anchor": "{p}1", "candidate": "{p}5", "verdict": "same"}}]}}
В каждом checks повтори id персонажа и номер главного кандидата из группы. Проверяй по примерам, не по памяти о книге.

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
        bits.append("обращения при имени: " + ", ".join(t for t, _ in c.titles.most_common(3)))
    near = [ids[(c.scope, k)] for k, _ in c.together.most_common(8) if (c.scope, k) in ids][:4]
    if near:
        bits.append("рядом: " + ", ".join(near))
    line = " | ".join(bits)
    for e in c.examples[:3]:
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
        if c.count >= args.min_count or c.speaker > 0 or (c.kind == "name" and (c.roles.get("Name") or c.roles.get("Patr"))):
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


def read_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save(folder: str, name: str, data, compact: bool = False) -> None:
    import tempfile
    target = os.path.join(folder, name)
    fd, temp = tempfile.mkstemp(prefix=".mytts-", suffix=".tmp", dir=folder)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, **({"separators": (",", ":")} if compact else {"indent": 1}))
        os.replace(temp, target)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


# ---------------------------------------------------------------- LLM: Ollama Cloud или API DeepSeek

HERE = os.path.dirname(os.path.abspath(__file__))
PROVIDERS = {
    "ollama": {"endpoint": "https://ollama.com", "model": "deepseek-v4.1-flash", "key": "OLLAMA_API_KEY"},
    "deepseek": {"endpoint": "https://api.deepseek.com", "model": "deepseek-flash", "key": "DEEPSEEK_API_KEY"},
}
# Предел ответа по умолчанию: с размышлением / без его.
MAX_TOKENS = {
    "ollama": {"think": 80000, "plain": 16000},
    "deepseek": {"think": 64000, "plain": 16000},
}


class LLMError(RuntimeError):
    """Ошибка сервиса LLM: код HTTP, текст ответа и (для 429) сколько секунд ждать до повтора."""

    def __init__(self, status: int, message: str, retry_after: str | None = None, max_output_tokens: int | None = None) -> None:
        super().__init__(f"HTTP {status}: {message}")
        self.status = status
        self.retry_after = retry_after
        self.max_output_tokens = max_output_tokens

    def wait_seconds(self, default: float) -> float:
        """Сколько ждать до повтора: заголовок Retry-After сервиса (не больше двух минут), иначе заданное."""
        if not self.retry_after:
            return default
        try:
            return min(120.0, max(1.0, float(self.retry_after)))
        except ValueError:
            return default


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


_THINKING_CONTROLS: dict[tuple, list] = {}
_MODEL_LIMITS: dict[tuple, int] = {}


def thinking_control(values: list, enabled: bool):
    if values == [False]:
        return False
    if values == [True]:
        return True
    if any(v is enabled for v in values):
        return enabled
    levels = [v for v in values if isinstance(v, str)]
    order = ("high", "max", "medium", "low", "minimal") if enabled else ("minimal", "low", "medium", "high", "max")
    if levels:
        return next((v for v in order if v in levels), levels[0])
    return enabled


def ollama_thinking(endpoint: str, model: str, key: str, enabled: bool):
    cache_key = (endpoint.rstrip("/"), model)
    if cache_key not in _THINKING_CONTROLS:
        req = urllib.request.Request(endpoint.rstrip("/") + "/api/show", data=json.dumps({"model":model}).encode(),
            headers={"Content-Type":"application/json", "Authorization":"Bearer " + key})
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.load(response)
            thinking = data.get("thinking")
            values = thinking.get("values") if isinstance(thinking, dict) else None
            _THINKING_CONTROLS[cache_key] = values if isinstance(values, list) else []
        except urllib.error.HTTPError as e:
            status = e.code
            e.close()
            if status not in (404, 405):
                raise LLMError(status, "сервер отклонил сведения о модели") from None
            _THINKING_CONTROLS[cache_key] = []
    values = _THINKING_CONTROLS[cache_key]
    if enabled and values and thinking_control(values, True) is False:
        raise ValueError("Выбранная модель не поддерживает размышление")
    return thinking_control(values, enabled)


def request_chat(provider: str, endpoint: str, model: str, key: str, prompt: str, think: bool, max_tokens: int, timeout: int) -> dict:
    """Один запрос. Возвращает content, thinking (длина), причину остановки и токены в общем виде."""
    if provider == "deepseek":
        # Оригинальный API DeepSeek (api.deepseek.com): размышление — thinking, ответ — в reasoning_content.
        url = endpoint.rstrip("/") + "/chat/completions"
        body = {"model": model, "stream": False, "max_tokens": max_tokens, "temperature": 0,
                "thinking": {"type": "enabled" if think else "disabled"},
                "messages": [{"role": "user", "content": prompt}]}
        if think:
            body["reasoning_effort"] = "high"
    else:
        url = endpoint.rstrip("/") + "/api/chat"
        body = {"model": model, "stream": False, "think": ollama_thinking(endpoint, model, key, think), "options": {"temperature": 0, "num_predict": max_tokens},
                "messages": [{"role": "user", "content": prompt}]}
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer " + key})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            data = json.loads(response.read())
    except urllib.error.HTTPError as e:  # текст ошибки сервиса, без ключа
        retry = e.headers.get("retry-after") if e.headers else None
        # Keep only the numeric limit; never retain/log arbitrary response bodies.
        try:
            message = json.loads(e.read()).get("error", "")
            match = re.search(r"maximum output tokens \((\d+)\)", message) if isinstance(message, str) else None
            cap = int(match.group(1)) if match else None
        except (ValueError, TypeError):
            cap = None
        finally:
            e.close()
        raise LLMError(e.code, "сервер отклонил запрос", retry, cap) from None
    if provider == "deepseek":
        choice = data["choices"][0]
        message = choice.get("message", {})
        usage = data.get("usage", {})
        thinking = message.get("reasoning_content") or message.get("reasoning") or ""
        return {"content": message.get("content") or "", "thinking_chars": len(thinking),
                "done_reason": choice.get("finish_reason"), "prompt_tokens": usage.get("prompt_tokens"),
                "output_tokens": usage.get("completion_tokens")}
    message = data.get("message", {})
    return {"content": message.get("content", ""), "thinking_chars": len(message.get("thinking") or ""),
            "thinking_control": body["think"], "done_reason": data.get("done_reason"), "prompt_tokens": data.get("prompt_eval_count"),
            "output_tokens": data.get("eval_count")}


def llm(args) -> None:
    load_env()
    preset = PROVIDERS[args.provider]
    args.model = args.model or preset["model"]
    args.endpoint = args.endpoint or preset["endpoint"]
    key = os.environ.get(preset["key"], "")
    if not key:
        sys.exit(f"Нет ключа: задайте {preset['key']} в переменной окружения или в файле .env (см. README.md)")
    request = read_json(os.path.join(args.dir, "llm_request.json"))
    by_id = {c["id"]: c for c in read_json(os.path.join(args.dir, "candidates.json"))["candidates"]}
    folder = os.path.join(args.dir, "answers")
    os.makedirs(folder, exist_ok=True)

    input_sha = artifact_identity(args.dir)

    def identity(prompt: str) -> str:
        return cache_fingerprint(args.provider, args.endpoint, args.model, args.think, prompt,
                                 args.max_tokens or MAX_TOKENS[args.provider]["think" if args.think else "plain"], input_sha)

    def cached(prompt: str, name: str, verify: bool = False) -> bool:
        if args.redo:
            return False
        answer = load_answer(os.path.join(folder, name + ".json"))
        meta = load_answer(os.path.join(folder, name + ".meta.json"))
        return bool(valid_response(answer, verify) and meta and meta.get("request_sha256") == identity(prompt)
                    and meta.get("done_reason") != "length")

    def chat(prompt: str, name: str, verify: bool = False) -> str:
        requested_limit = args.max_tokens or MAX_TOKENS[args.provider]["think" if args.think else "plain"]
        model_key = (args.provider, args.endpoint, args.model)
        limit = min(requested_limit, _MODEL_LIMITS.get(model_key, requested_limit))
        delay = 2.0
        for attempt in range(1, 4):
            started = time.time()
            try:
                reply = request_chat(args.provider, args.endpoint, args.model, key, prompt, args.think, limit, args.timeout)
            except LLMError as e:
                if attempt < 3 and e.status == 400 and e.max_output_tokens and 0 < e.max_output_tokens < limit:
                    limit = e.max_output_tokens
                    _MODEL_LIMITS[model_key] = limit
                    print(f"  [{name}] API ограничивает ответ {limit} токенами; повторяем с допустимым пределом", flush=True)
                    continue
                if attempt >= 3 or e.status not in (429, 500, 502, 503, 504):
                    raise
                delay = e.wait_seconds(delay * 2)
                print(f"  [{name}] HTTP {e.status}; ждём {delay:.0f} с и повторяем ({attempt + 1}/3)", flush=True)
                time.sleep(delay)
                continue
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                if attempt >= 3:
                    raise
                print(f"  [{name}] временный сетевой сбой {type(e).__name__}; ждём {delay:.0f} с и повторяем ({attempt + 1}/3)", flush=True)
                time.sleep(delay)
                delay = min(120, delay * 2)
                continue
            content = reply.pop("content")
            answer = parse_answer(content)
            if reply.get("done_reason") == "length" or not valid_response(answer, verify):
                if attempt < 3:
                    print(f"  [{name}] неполный или некорректный ответ; повторяем ({attempt + 1}/3)", flush=True)
                    continue
                raise ValueError("LLM не вернула полный JSON нужного формата; ответ не сохранён")
            meta = dict(reply, provider=args.provider, model=args.model, content_chars=len(content),
                        request_sha256=identity(prompt), thinking=args.think, endpoint=args.endpoint, max_tokens=requested_limit, effective_max_tokens=limit, input_sha256=input_sha,
                        seconds=round(time.time() - started, 1), attempt=attempt)
            # Never expose a partly written reply as a completed cache entry.
            save(folder, name + ".json", answer)
            save(folder, name + ".meta.json", meta)
            return (f"{meta['seconds']} с, ответ {meta['content_chars']} знаков, размышление {meta['thinking_chars']}, "
                    f"токенов {meta['prompt_tokens']}→{meta['output_tokens']}, {meta['done_reason']}")
        raise RuntimeError("Не удалось получить ответ LLM")

    def ask(r: dict) -> str:
        name = r["name"]
        line = "основной ответ из кэша" if cached(r["prompt"], name) else chat(r["prompt"], name)
        answer = load_answer(os.path.join(folder, name + ".json"))
        prompt = verify_prompt(r, answer, by_id, request["scope"] == "section") if answer else None
        if not prompt:
            return line + "; проверять нечего"
        verification = "из кэша" if cached(prompt, name + ".verify", True) else chat(prompt, name + ".verify", True)
        checked = load_answer(os.path.join(folder, name + ".verify.json"))
        if not verification_complete(checked, answer, r, by_id):
            # A well-formed JSON may still omit/repeat a check or name the wrong anchor.
            # Do not cache that as a complete verification, nor report a successful export.
            if os.path.exists(os.path.join(folder, name + ".verify.meta.json")):
                os.unlink(os.path.join(folder, name + ".verify.meta.json"))
            raise ValueError("Проверка склеек неполна или относится к другой группе; повторите запуск")
        return line + "; проверка: " + verification

    started = time.time()
    todo = request["requests"]
    print(f"Разделов {len(todo)}, {args.provider}: {args.model}, "
          f"размышление {'вкл' if args.think else 'выкл'}, одновременно {args.parallel}; готовый кэш переиспользуется", flush=True)
    failures = []
    with concurrent.futures.ThreadPoolExecutor(args.parallel) as pool:
        for r, line in zip(todo, pool.map(lambda r: safe(ask, r), todo)):
            print(f"  [{r['name']}] {r['title']}: {line}", flush=True)
            if line.startswith("ошибка:"):
                failures.append(r["name"])
    if failures:
        raise RuntimeError("Не завершены запросы LLM: " + ", ".join(failures) + ". Повторите запуск; готовые ответы сохранены.")
    print(f"Готово за {time.time() - started:.1f} с")


def safe(fn, r) -> str:
    try:
        return fn(r)
    except Exception as e:  # сетевой сбой одного рассказа не останавливает остальные
        return f"ошибка: {type(e).__name__}: {str(e)[:160]}"


def character_refs(raw: dict, r: dict, candidates: dict) -> list[str]:
    refs = raw.get("candidates", [])
    if not isinstance(refs, list):
        return []
    own = set(r["candidates"])
    refs = list(dict.fromkeys(ref for ref in refs if isinstance(ref, str) and ref in own and ref in candidates))
    # Names anchor a merge, never a generic title/family. Most complete names first.
    return sorted(refs, key=lambda ref: (candidates[ref]["kind"] != "name",
        -len(candidates[ref]["key"].split()), -candidates[ref]["count"], ref))


def verification_complete(checked: dict | None, answer: dict, r: dict, candidates: dict) -> bool:
    if not valid_response(checked, True):
        return False
    expected = []
    for raw in answer["characters"]:
        refs = character_refs(raw, r, candidates)
        expected.extend((raw["id"], refs[0], ref) for ref in refs[1:])
    actual = [(check["character"], check["anchor"], check["candidate"]) for check in checked["checks"]]
    return collections.Counter(actual) == collections.Counter(expected) and len(actual) == len(set(actual))


def verification_verdicts(checked: dict | None, answer: dict | None, r: dict, candidates: dict) -> dict:
    verdicts = {}
    expected = {}
    raw_characters = (answer or {}).get("characters", [])
    for raw in raw_characters if isinstance(raw_characters, list) else []:
        if not isinstance(raw, dict):
            continue
        refs = character_refs(raw, r, candidates)
        for ref in refs[1:]:
            expected.setdefault(ref, []).append((str(raw.get("id", "")), refs[0]))
    checks = (checked or {}).get("checks", [])
    if not isinstance(checks, list):
        return verdicts
    seen = collections.Counter()
    for check in checks:
        if not isinstance(check, dict) or not isinstance(check.get("candidate"), str):
            continue
        ref = check["candidate"]
        # Legacy unbound verdicts are intentionally not used for a new merge.
        pair = (check.get("character"), check.get("anchor"))
        if not all(isinstance(x, str) for x in pair):
            continue
        if pair not in expected.get(ref, []):
            continue
        key = (pair[0], pair[1], ref)
        seen[key] += 1
        verdicts[key] = check.get("verdict") if seen[key] == 1 else "unsure"
    return verdicts


def verify_prompt(r: dict, answer: dict, by_id: dict, collection: bool) -> str | None:
    """Второй запрос: каждую склейку подтверждает отдельный ответ «тот же / другой / не уверена»."""
    groups = []
    raw_characters = answer.get("characters", [])
    for raw in raw_characters if isinstance(raw_characters, list) else []:
        if not isinstance(raw, dict):
            continue
        refs = character_refs(raw, r, by_id)
        if len(refs) < 2:
            continue
        lines = [f"Группа id={raw.get('id', '')}. Главный: {describe(by_id[refs[0]])}"]
        lines += [f"  проверить {describe(by_id[ref])}" for ref in refs[1:]]
        groups.append("\n".join(lines))
    if not groups:
        return None
    what = f"рассказе «{r['title']}»" if collection else f"книге «{r['title']}»"
    prefix = r["candidates"][0].rstrip("0123456789")
    return VERIFY.format(what=what, p=prefix, groups="\n\n".join(groups))


def describe(c: dict) -> str:
    text = f"{c['id']} {c['display']} (упоминаний {c['count']}, род {c['gender']}, формы: {', '.join(list(c['forms'])[:4])})"
    for e in c["examples"][:3]:
        text += f"\n      пример: {e}"
    return text


# ---------------------------------------------------------------- проверка ответа

def artifact_identity(folder: str) -> str:
    digest = hashlib.sha256()
    for name in ("candidates.json", "llm_request.json", "book_index.json"):
        path = os.path.join(folder, name)
        digest.update(name.encode())
        if os.path.isfile(path):
            with open(path, "rb") as f:
                digest.update(f.read())
    return digest.hexdigest()


def metadata_matches(meta: dict | None, prompt: str, folder: str) -> bool:
    if not meta or not all(k in meta for k in ("provider", "endpoint", "model", "thinking", "max_tokens", "request_sha256", "input_sha256", "done_reason")):
        return False
    if not all(isinstance(meta[k], str) for k in ("provider", "endpoint", "model", "request_sha256", "input_sha256", "done_reason")) or not isinstance(meta["thinking"], bool) or not isinstance(meta["max_tokens"], int):
        return False
    source = artifact_identity(folder)
    return meta["input_sha256"] == source and meta["done_reason"] != "length" and meta["request_sha256"] == cache_fingerprint(
        meta["provider"], meta["endpoint"], meta["model"], meta["thinking"], prompt, meta["max_tokens"], source)


def cache_fingerprint(provider: str, endpoint: str, model: str, think: bool, prompt: str, max_tokens: int, input_sha256: str = "") -> str:
    return hashlib.sha256(json.dumps(["cast-verify-v2", provider, endpoint.rstrip("/"), model,
                                     think, max_tokens, prompt, input_sha256], ensure_ascii=False).encode()).hexdigest()


def parse_answer(text: str) -> dict | None:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("повторный ключ JSON")
            result[key] = value
        return result
    start, end = text.find("{"), text.rfind("}")
    try:
        value = json.loads(text[start:end + 1], object_pairs_hook=unique) if 0 <= start < end else None
        return value if isinstance(value, dict) else None
    except (ValueError, TypeError):
        return None


def valid_response(answer: dict | None, verify: bool = False) -> bool:
    if not isinstance(answer, dict):
        return False
    if verify:
        return isinstance(answer.get("checks"), list) and all(isinstance(c, dict) and
            all(isinstance(c.get(k), str) for k in ("character", "anchor", "candidate")) and
            c.get("verdict") in ("same", "different", "unsure") for c in answer["checks"])
    return isinstance(answer.get("characters"), list) and isinstance(answer.get("other"), list) and all(
        isinstance(c, dict) and isinstance(c.get("id"), str) and isinstance(c.get("name"), str) and
        c.get("gender") in ("m", "f", "?") and isinstance(c.get("candidates"), list) and
        all(isinstance(ref, str) for ref in c["candidates"]) for c in answer["characters"]) and all(
        isinstance(ref, str) for ref in answer["other"])


def load_answer(path: str) -> dict | None:
    try:
        with open(path, encoding="utf-8") as f:
            return parse_answer(f.read())
    except (OSError, UnicodeError):
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
    raw_characters = answer.get("characters", [])
    if not isinstance(raw_characters, list):
        raw_characters = []
        problems.append("characters должен быть списком — все кандидаты в прочих")
    ownership = collections.Counter(ref for raw in raw_characters if isinstance(raw, dict)
        for ref in (raw.get("candidates", []) if isinstance(raw.get("candidates", []), list) else [])
        if isinstance(ref, str) and ref in own_ids)
    explicit_other = answer.get("other", [])
    if not isinstance(explicit_other, list):
        explicit_other = []
    ownership.update(ref for ref in explicit_other if isinstance(ref, str) and ref in own_ids)
    conflicts = {ref for ref, count in ownership.items() if count > 1}
    if conflicts:
        problems.append("повторные кандидаты → прочие: " + ", ".join(sorted(conflicts)))
    for raw in raw_characters:
        if not isinstance(raw, dict):
            problems.append("персонаж должен быть объектом — пропущен")
            continue
        cid = re.sub(r"[^a-z0-9_]", "_", str(raw.get("id", "")).lower()).strip("_")
        cid = prefix + cid if cid else ""
        if not cid or cid.endswith(("author", "other")) or any(ch["id"] == cid for ch in characters):
            problems.append(f"идентификатор «{raw.get('id')}» пустой, служебный или повторяется — персонаж пропущен")
            continue
        own = []
        gender = raw.get("gender") if raw.get("gender") in ("m", "f", "?") else "?"
        refs = character_refs(raw, r, candidates)
        if not refs:
            problems.append(f"{cid}: нет допустимых кандидатов — пропущен")
            continue
        anchor = refs[0]
        for ref in refs:
            candidate = candidates[ref]
            if ref in conflicts:
                dropped.append(f"{ref} {candidate['display']} → прочие: несколько владельцев")
            elif candidate["kind"] == "family":
                dropped.append(f"{ref} {candidate['display']} → прочие: семья, не один человек")
            elif gender in ("m", "f") and candidate["gender"] in ("m", "f") and candidate["gender"] != gender \
                    and candidate.get("gender_source") in ("verb", "Patr", "Title"):
                dropped.append(f"{ref} {candidate['display']} → прочие: род {candidate['gender']}, у {cid} {gender}")
            elif ref != anchor and (anchor not in own or not verdicts or verdicts.get((str(raw.get("id", "")), anchor, ref)) != "same"):
                dropped.append(f"{ref} {candidate['display']} → прочие: склейка с {anchor} не подтверждена для {cid}")
            else:
                used[ref] = cid
                own.append(ref)
        if not own:
            problems.append(f"{cid}: нет ни одного кандидата — пропущен")
            continue
        characters.append({"id": cid, "name": str(raw.get("name") or ""), "gender": gender, "candidates": own})
    # Any named candidate can contradict a title, even if the LLM omitted that person.
    for ch in characters:
        for ref in [ref for ref in ch["candidates"] if candidates[ref]["kind"] == "title"]:
            word = candidates[ref]["key"]
            per = {other["id"]: sum(candidates[x]["titles"].get(word, 0) for x in other["candidates"] if x != ref) for other in characters}
            unassigned = sum(candidates[x]["titles"].get(word, 0) for x in own_ids
                             if candidates[x]["kind"] == "name" and used.get(x, "other") == "other")
            total = sum(per.values()) + unassigned
            if (total > 0 and per[ch["id"]] < total) or (total == 0 and len(ch["candidates"]) == 1):
                ch["candidates"].remove(ref)
                used[ref] = "other"
                other_owner = max(per, key=per.get) if total else None
                dropped.append(f"{ref} {candidates[ref]['display']} → прочие: обращение связано с именем {ch['id']} "
                               f"{per[ch['id']]} из {total} раз" + (f" (чаще {other_owner})" if other_owner and other_owner != ch["id"] else ""))
    characters = [ch for ch in characters if ch["candidates"]]
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
    for ref in explicit_other:
        if isinstance(ref, str) and ref in own_ids and ref not in used:
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
        for form in candidates[ref]["forms"]:
            alias[form.lower().replace("ё", "е")].add("other")
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
    data = read_json(os.path.join(args.dir, "candidates.json"))
    request = read_json(os.path.join(args.dir, "llm_request.json"))
    candidates = {c["id"]: c for c in data["candidates"]}
    collection = request["scope"] == "section"
    casts = []
    for r in request["requests"]:
        answer = load_answer(os.path.join(args.dir, "answers", r["name"] + ".json"))
        main_meta = load_answer(os.path.join(args.dir, "answers", r["name"] + ".meta.json"))
        if not metadata_matches(main_meta, r["prompt"], args.dir) or not valid_response(answer):
            raise ValueError(f"[{r['name']}] ответ устарел или не проверен: выполните llm перед apply")
        checked = load_answer(os.path.join(args.dir, "answers", r["name"] + ".verify.json"))
        check_prompt = verify_prompt(r, answer, candidates, collection)
        check_meta = load_answer(os.path.join(args.dir, "answers", r["name"] + ".verify.meta.json"))
        if check_prompt and not metadata_matches(check_meta, check_prompt, args.dir):
            checked = None
        verdicts = verification_verdicts(checked, answer, r, candidates)
        if verdicts is None and answer and any(len(ch.get("candidates", [])) > 1 for ch in answer.get("characters", [])):
            verdicts = {}  # склейки без второй проверки не принимаются
        casts.append(build_cast(r, answer, verdicts, candidates, r["name"] + "." if collection else ""))
    section_cast = {sid: i for i, cast in enumerate(casts) for sid in cast["sections"]}
    save(args.dir, "cast.json", {
        "book": data["book"], "scope": request["scope"], "input_sha256": artifact_identity(args.dir), "narrator": "author", "others": "other",
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
    cast = read_json(os.path.join(args.dir, "cast.json"))
    if cast.get("input_sha256") != artifact_identity(args.dir):
        raise ValueError("cast.json от другого извлечения: повторите llm и apply")
    candidates = {c["id"]: c for c in read_json(os.path.join(args.dir, "candidates.json"))["candidates"]}
    spec = read_json(args.voices)
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
            if not main or not pool.get(ch["gender"]):
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
    cast = read_json(os.path.join(args.dir, "cast.json"))
    if cast.get("input_sha256") != artifact_identity(args.dir):
        raise ValueError("cast.json от другого извлечения: повторите llm и apply")
    index = read_json(os.path.join(args.dir, "book_index.json"))
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
        ambiguous = set(group.get("ambiguous", {}))
        for item in characters:
            item["forms"] = [form for form in item["forms"] if form.lower().replace("ё", "е") not in ambiguous]
        # Unrecognized aliases stay explicitly in «прочие», including their case forms.
        candidates = {c["id"]: c for c in read_json(os.path.join(args.dir, "candidates.json"))["candidates"]}
        other = []
        for entry in group["other"]:
            candidate = candidates[entry["candidate"]]
            other += [entry["display"], candidate["key"]] + list(candidate["forms"])[:args.max_forms]
        for ch in group["characters"]:
            if ch.get("role") == "other" and not args.keep_other:
                other += [ch["name"]] + ch["forms"][:args.max_forms]
        other = list(dict.fromkeys(other + sorted(ambiguous)))
        casts.append({"sections": group["sections"], "characters": characters, "other": other})
    data = {
        "format": "mytts-book", "version": 1, "book": index["book"], "scope": cast["scope"],
        "voice_model": cast.get("voice_model", ""),
        "sections": [{"id": s["id"], "title": s["title"], "cast": s.get("cast")} for s in cast["sections"]],
        "casts": casts, "fingerprint": index["fingerprint"], "fingerprints": dict(by_section),
    }
    target = args.output or os.path.join(args.dir, re.sub(r"[^\w.-]+", "_", index["book"]["title"]) + ".mytts-book")
    save(os.path.dirname(os.path.abspath(target)), os.path.basename(target), data, compact=True)
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
    parser.add_argument("--provider", choices=sorted(PROVIDERS), default="ollama", help="ollama (Ollama Cloud) или deepseek (API DeepSeek)")
    parser.add_argument("--model", help="по умолчанию: ollama — deepseek-v4.1-flash, deepseek — deepseek-flash")
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
