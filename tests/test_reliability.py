import collections
import contextlib
import copy
import io
import json
import os
import random
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import mytts_book as b


def candidate(key, kind="name", gender="m", source="Patr", count=3, forms=None, titles=None):
    return dict(key=key, display=key, kind=kind, gender=gender, gender_source=source,
                count=count, speaker=0, forms=forms or {key: count}, titles=titles or {}, examples=[])


def character(cid, refs, gender="m"):
    return dict(id=cid, name="", gender=gender, candidates=refs)


class CastTests(unittest.TestCase):
    def setUp(self):
        self.c = {"c1": candidate("николай"), "c2": candidate("николай петрович"),
                  "c3": candidate("петр иванович"), "c4": candidate("генерал", "title", source="Title"),
                  "c5": candidate("семья иволгин", "family", "?", None)}
        self.r = dict(candidates=list(self.c), sections=["s1"], title="Fixture")

    def cast(self, characters, other=None, checks=None):
        a = dict(characters=characters, other=other or [])
        v = b.verification_verdicts(dict(checks=checks or []), a, self.r, self.c)
        return b.build_cast(self.r, a, v, self.c, "")

    def test_unverified_merge_is_other(self):
        result = self.cast([character("n", ["c1", "c2"])])
        self.assertEqual(["c2"], result["characters"][0]["candidates"])
        self.assertIn("c1", {x["candidate"] for x in result["other"]})

    def test_bound_confirmation_accepts_merge(self):
        result = self.cast([character("n", ["c1", "c2"])], checks=[dict(character="n", anchor="c2", candidate="c1", verdict="same")])
        self.assertEqual(["c2", "c1"], result["characters"][0]["candidates"])

    def test_confirmation_for_different_anchor_is_rejected(self):
        result = self.cast([character("n", ["c1", "c2"])], checks=[dict(character="n", anchor="c3", candidate="c1", verdict="same")])
        self.assertEqual(["c2"], result["characters"][0]["candidates"])

    def test_legacy_unbound_confirmation_is_rejected(self):
        result = self.cast([character("n", ["c1", "c2"])], checks=[dict(candidate="c1", verdict="same")])
        self.assertEqual(["c2"], result["characters"][0]["candidates"])

    def test_duplicate_candidate_does_not_go_to_first_owner(self):
        result = self.cast([character("a", ["c1"]), character("b", ["c1"])])
        self.assertEqual([], result["characters"])
        self.assertIn("c1", {x["candidate"] for x in result["other"]})

    def test_conflicting_other_is_not_assigned(self):
        result = self.cast([character("a", ["c1"])], other=["c1"])
        self.assertEqual([], result["characters"])

    def test_invalid_primary_cannot_promote_unverified_alias(self):
        result = self.cast([character("a", ["c2", "c1"], "f")])
        self.assertEqual([], result["characters"])

    def test_gender_checked_on_primary(self):
        self.assertEqual([], self.cast([character("a", ["c2"], "f")])["characters"])

    def test_family_is_never_a_person(self):
        self.assertEqual([], self.cast([character("family", ["c5"], "?")])["characters"])

    def test_generic_title_is_not_a_standalone_person(self):
        self.assertEqual([], self.cast([character("general", ["c4"])])["characters"])

    def test_implicit_title_can_be_verified(self):
        result = self.cast([character("n", ["c4", "c2"])], checks=[dict(character="n", anchor="c2", candidate="c4", verdict="same")])
        self.assertIn("c4", result["characters"][0]["candidates"])

    def test_title_used_for_multiple_people_is_other(self):
        self.c["c2"]["titles"] = {"генерал": 1}
        self.c["c3"]["titles"] = {"генерал": 2}
        result = self.cast([character("n", ["c4", "c2"]), character("p", ["c3"])], checks=[dict(character="n", anchor="c2", candidate="c4", verdict="same")])
        self.assertNotIn("c4", result["characters"][0]["candidates"])

    def test_repeated_verdict_is_not_last_wins(self):
        check = dict(character="n", anchor="c2", candidate="c1", verdict="same")
        result = self.cast([character("n", ["c1", "c2"])], checks=[check, check])
        self.assertEqual(["c2"], result["characters"][0]["candidates"])

    def test_title_also_used_for_unrecognized_person_is_other(self):
        self.c["c2"]["titles"] = {"генерал": 5}
        self.c["c3"]["titles"] = {"генерал": 1}
        result = self.cast([character("n", ["c4", "c2"])], other=["c3"],
                           checks=[dict(character="n", anchor="c2", candidate="c4", verdict="same")])
        self.assertNotIn("c4", result["characters"][0]["candidates"])
        self.assertIn("c4", {x["candidate"] for x in result["other"]})

    def test_unrecognized_candidate_forms_block_unique_alias(self):
        self.c["c3"]["forms"] = {"Николай": 1}
        result = self.cast([character("n", ["c1"])], other=["c3"])
        self.assertNotIn("николай", result["alias_index"])
        self.assertEqual(["n", "other"], result["ambiguous"]["николай"])

    def test_malformed_candidates_do_not_crash(self):
        result = self.cast([character("n", [[], {}, "c1"])])
        self.assertEqual(["c1"], result["characters"][0]["candidates"])

    def test_malformed_top_level_goes_to_other(self):
        result = b.build_cast(self.r, {"characters": None, "other": {}}, {}, self.c, "")
        self.assertEqual([], result["characters"])
        self.assertEqual(len(self.c), len(result["other"]))

    def test_partition_with_five_hundred_noisy_answers(self):
        rng=random.Random(20261008)
        for _ in range(500):
            raw=[]
            for n in range(rng.randrange(5)):
                refs=[rng.choice(list(self.c)+["invalid",None,{}]) for _ in range(rng.randrange(7))]
                raw.append(character("person"+str(n),refs,rng.choice(["m","f","?"])))
            other=[rng.choice(list(self.c)+["invalid",None]) for _ in range(rng.randrange(5))]
            result=self.cast(raw,other=other)
            owners=[ref for ch in result["characters"] for ref in ch["candidates"]]+[item["candidate"] for item in result["other"]]
            self.assertCountEqual(self.r["candidates"],owners)
            self.assertEqual(len(owners),len(set(owners)))

    def test_every_candidate_has_exactly_one_owner(self):
        result = self.cast([character("n", ["c1", "c2"]), character("p", ["c3"])], checks=[dict(character="n", anchor="c2", candidate="c1", verdict="same")])
        assigned = [ref for ch in result["characters"] for ref in ch["candidates"]] + [o["candidate"] for o in result["other"]]
        self.assertCountEqual(self.r["candidates"], assigned)
        self.assertEqual(len(assigned), len(set(assigned)))

    def test_bare_surname_shared_by_named_relatives_is_other(self):
        self.c["c1"] = candidate("иволгин",count=20)
        self.c["c1"]["roles"] = {"Surn":20}
        self.c["c2"] = candidate("ардалион александрович")
        self.c["c3"] = candidate("гаврила ардалионович иволгин", count=10)
        result = self.cast([character("father",["c1","c2"]),character("son",["c3"])],
                           checks=[dict(character="father",anchor="c2",candidate="c1",verdict="same")])
        self.assertEqual(["c2"], result["characters"][0]["candidates"])
        self.assertIn("c1",{x["candidate"] for x in result["other"]})

    def test_surname_of_rarely_named_relative_stays_with_main_character(self):
        # «Рогожин» 26 раз; отец «Семен Парфенович Рогожин» назван полностью дважды.
        self.c["c1"] = candidate("рогожин", count=26)
        self.c["c1"]["roles"] = {"Surn": 26}
        self.c["c2"] = candidate("парфен")
        self.c["c3"] = candidate("семен парфенович рогожин", count=2)
        result = self.cast([character("son", ["c1", "c2"]), character("father", ["c3"])],
                           checks=[dict(character="son", anchor="c2", candidate="c1", verdict="same")])
        self.assertIn("c1", result["characters"][0]["candidates"])

    def test_ambiguous_surname_anchor_cannot_join_other_aliases(self):
        self.c["c1"] = candidate("иволгин",count=20)
        self.c["c1"]["roles"] = {"Surn":20}
        self.c["c2"] = candidate("ардалион",count=2)
        self.c["c3"] = candidate("гаврила ардалионович иволгин", count=10)
        result = self.cast([character("father",["c1","c2"]),character("son",["c3"])],
                           checks=[dict(character="father",anchor="c1",candidate="c2",verdict="same")])
        self.assertEqual(["son"],[c["id"] for c in result["characters"]])
        self.assertTrue({"c1","c2"}<={x["candidate"] for x in result["other"]})


    def test_unsure_merge_accepted_when_short_name_is_part_of_full_name(self):
        result = self.cast([character("n", ["c1", "c2"])], checks=[dict(character="n", anchor="c2", candidate="c1", verdict="unsure")])
        self.assertEqual(["c2", "c1"], result["characters"][0]["candidates"])

    def test_unsure_merge_rejected_when_short_name_fits_another_person(self):
        self.c["c3"] = candidate("николай андреевич")
        result = self.cast([character("n", ["c1", "c2"])], checks=[dict(character="n", anchor="c2", candidate="c1", verdict="unsure")])
        self.assertEqual(["c2"], result["characters"][0]["candidates"])

    def test_different_verdict_always_rejects_merge(self):
        result = self.cast([character("n", ["c1", "c2"])], checks=[dict(character="n", anchor="c2", candidate="c1", verdict="different")])
        self.assertEqual(["c2"], result["characters"][0]["candidates"])

    def label_cast(self, votes):
        self.c["c4"]["contexts"] = ["…"] * sum(votes.values())
        a = dict(characters=[character("n", ["c2", "c4"]), character("p", ["c3"])], other=[])
        v = {("n", "c2", "c4"): "same"}
        return b.build_cast(self.r, a, v, self.c, "", {("n", "c4"): collections.Counter(votes)})

    def test_title_kept_when_passages_mostly_name_owner(self):
        # «генерал» иногда называет другого человека, но почти везде — этого.
        self.c["c3"]["titles"] = {"генерал": 3}
        result = self.label_cast({"n": 14, "p": 1, "unsure": 1})
        self.assertIn("c4", result["characters"][0]["candidates"])

    def test_title_dropped_when_passages_split_between_people(self):
        result = self.label_cast({"n": 9, "p": 6, "unsure": 1})
        self.assertNotIn("c4", result["characters"][0]["candidates"])
        self.assertIn("c4", {x["candidate"] for x in result["other"]})

    def test_title_dropped_when_passages_mostly_unclear(self):
        result = self.label_cast({"n": 3, "unsure": 13})
        self.assertNotIn("c4", result["characters"][0]["candidates"])


class CombineTests(unittest.TestCase):
    def setUp(self):
        self.c = {f"c{n}": candidate(f"имя{n}", count=10 - n) for n in range(1, 6)}
        self.r = dict(candidates=list(self.c))

    def test_majority_keeps_merge_and_drops_single_run_mistake(self):
        runs = [{"characters": [character("a", ["c1", "c2"]), character("b", ["c3"])], "other": ["c4", "c5"]},
                {"characters": [character("a", ["c1", "c2", "c3"])], "other": ["c4", "c5"]},
                {"characters": [character("a", ["c1", "c2"]), character("b", ["c3", "c4"])], "other": ["c5"]}]
        combined = b.combine_answers(runs, self.r, self.c)
        groups = {ch["id"]: ch["candidates"] for ch in combined["characters"]}
        self.assertEqual({"a": ["c1", "c2"], "b": ["c3"]}, groups)
        self.assertEqual(["c4", "c5"], combined["other"])

    def test_identifiers_are_unique(self):
        runs = [{"characters": [character("a", ["c1"]), character("a", ["c2"])], "other": []}] * 3
        ids = [ch["id"] for ch in b.combine_answers(runs, self.r, self.c)["characters"]]
        self.assertEqual(len(ids), len(set(ids)))


class ExtractorTests(unittest.TestCase):
    def extract(self, paragraphs):
        return b.Extractor().run(b.Book("Fixture", "", [{"id": "s1", "title": "Test"}], [(0,t) for t in paragraphs]), False)

    def test_epanchin_spouses_stay_separate(self):
        c = self.extract(["Генерал Епанчин вошёл. Генеральша Епанчина ответила. Генерал Епанчин ушёл."])
        self.assertIn((0, "епанчин"), c)
        self.assertIn((0, "епанчина"), c)
        self.assertEqual("m", b.gender_of(c[(0, "епанчин")]))
        self.assertEqual("f", b.gender_of(c[(0, "епанчина")]))

    def test_singular_ov_surname_is_a_person_not_a_family(self):
        c = self.extract(["Начинающий дизайнер Улямов боялся. Улямов взмахнул рукой. Мысль пришла Улямову."])
        self.assertEqual("name", c[(0, "улямов")].kind)
        self.assertEqual(3, c[(0, "улямов")].count)

    def test_plural_unknown_surname_is_a_family(self):
        c = self.extract(["Супруги Бобриковы пришли. Он встретил Бобриковых. Рядом с Бобриковыми."])
        self.assertEqual("family", c[(0, "семья бобриков")].kind)

    def test_diminutive_forms_join_most_frequent_base(self):
        text = ["Владя спал. Владя ел. Владя пил. Владя шёл.", "Пришёл Кирюха. Ушёл Кирюха. Сел Кирюха.",
                "Он позвал Владю. Он видел Кирюху. — Кирюх, иди! — Кирюх, стой!"]
        c = self.extract(text)
        self.assertEqual(5, c[(0, "владя")].count)
        self.assertEqual(4, c[(0, "кирюха")].count)

    def test_two_surnames_in_a_row_are_two_people(self):
        c = self.extract(["Рогожин пришёл. Рогожин ушёл.", "— А ты ступай за мной, строка, — сказал Рогожин Лебедеву."])
        self.assertNotIn((0, "рогожин лебедев"), c)
        self.assertEqual(1, c[(0, "рогожин")].speaker)

    def test_parenthetical_word_between_verb_and_speaker(self):
        c = self.extract(["Рогожин пришёл.", "— Эге! — действительно удивился, наконец, Рогожин; — да ведь он знает."])
        self.assertEqual(1, c[(0, "рогожин")].speaker)

    def test_descriptor_speaker_becomes_candidate(self):
        c = self.extract(["— Зябко? — спросил черномазый.", "— Куда же? — спросил черномазый.", "— Гм… — промычал удивленный лакей."])
        self.assertEqual(2, c[(0, "черномазый")].speaker)
        self.assertTrue(c[(0, "черномазый")].descriptor)
        self.assertEqual(1, c[(0, "лакей")].speaker)

    def test_plural_title_is_not_a_person(self):
        c = self.extract(["— Господа, — сказал он. Господа молчали."])
        self.assertNotIn((0, "господин"), c)

    def test_inflected_titles_are_detected(self):
        c = self.extract(["Он спросил князя. Потом он подошёл к князю и говорил с князем."])
        self.assertEqual(3, c[(0, "князь")].count)

    def test_unknown_adjectival_surname_is_normalized_from_text(self):
        c = self.extract(["Тоцкий вошёл. Тоцкий говорил. Он встретился с Тоцким. Он спорил с Тоцким."])
        self.assertIn((0, "тоцкий"), c)
        self.assertNotIn((0, "тоцким"), c)
        self.assertEqual(4, c[(0, "тоцкий")].count)

    def test_single_surname_with_adjectival_declension(self):
        c = self.extract(["Он говорил с Тоцким."])
        self.assertIn((0,"тоцкий"),c)

    def test_postposed_title_link_is_detected(self):
        c = self.extract(["Иван Петрович, отставной генерал, вошёл. Генерал Сидоров ушёл."])
        self.assertEqual(1, c[(0,"иван петрович")].titles["генерал"])
        self.assertEqual(1, c[(0,"сидоров")].titles["генерал"])

    def test_unseen_nominative_is_not_replaced_by_case_form(self):
        c = self.extract(["Он говорил с Афанасием Ивановичем и с Ардалионом Александровичем."])
        self.assertIn((0,"афанасий иванович"), c)
        self.assertIn((0,"ардалион александрович"), c)

    def test_surname_agrees_with_full_male_name(self):
        c = self.extract(["Он говорил о семействе Гаврилы Ардалионыча Иволгина."])
        self.assertIn((0,"гаврила ардалионыч иволгин"), c)

    def test_first_introduction_unknown_surname_is_found(self):
        c = self.extract(["— Да, я Рогожин, Парфен."])
        self.assertIn((0, "рогожин"), c)
        self.assertIn((0, "парфен"), c)

    def test_related_family_remains_plural(self):
        c = self.extract(["Он спросил про Рогожиных. Он спросил про Рогожиных."])
        self.assertEqual("family", c[(0, "семья рогожин")].kind)

    def test_internal_dash_does_not_make_mentioned_person_speaker(self):
        c = self.extract(["— Садитесь, — сказал генерал. — Нина Александровна и Варвара Александровна, — дамы, которых я уважаю. Нина Александровна примет вас, а я уже закончил разговор."])
        self.assertEqual(1, c[(0,"генерал")].speaker)
        self.assertEqual(0, c[(0,"нина александровна")].speaker)

    def test_speaker_is_near_reporting_verb_not_first_mentioned_person(self):
        c = self.extract(["— Здравствуйте, — к Ивану Петровичу подошёл Николай Павлович. — Я ждал."])
        self.assertEqual(1, c[(0,"николай павлович")].speaker)
        self.assertEqual(0, c[(0,"иван петрович")].speaker)

    def test_reporting_verb_with_adverbial_phrase(self):
        c = self.extract(["— Да, — отвечал в раздумьи чиновник. — Конечно, — тормошился чиновник."])
        self.assertEqual(2, c[(0,"чиновник")].speaker)

    def test_unknown_female_case_does_not_follow_more_frequent_male_surname(self):
        c = self.extract(["Генерал Епанчин вошёл. " * 12 +
                          "Генеральша Епанчина ответила. Он писал генеральше Епанчиной."])
        self.assertEqual(12, c[(0,"епанчин")].count)
        self.assertNotIn("Епанчиной", c[(0,"епанчин")].forms)
        self.assertEqual(2, c[(0,"епанчина")].count)
        self.assertIn("Епанчиной", c[(0,"епанчина")].forms)

    def test_unknown_surname_inherits_gender_from_patronymic(self):
        c = self.extract(["Генерал Епанчин вошёл. " * 12 +
                          "Генеральша Епанчина ответила. Он писал Лизавете Прокофьевне Епанчиной."])
        self.assertIn((0,"лизавета прокофьевна епанчина"), c)
        self.assertNotIn((0,"лизавета прокофьевна епанчин"), c)

    def test_addressee_and_speaker_are_two_people(self):
        c = self.extract(["— Пора, — объявила Евгению Павловичу Лизавета Прокофьевна."])
        self.assertIn((0,"евгений павлович"), c)
        self.assertIn((0,"лизавета прокофьевна"), c)
        self.assertNotIn((0,"евгений павлович лизавета прокофьевна"), c)
        self.assertEqual(1, c[(0,"лизавета прокофьевна")].speaker)

    def test_adjacent_same_gender_full_names_are_separate(self):
        c = self.extract(["Вошли Иван Петрович Николай Павлович."])
        self.assertIn((0,"иван петрович"), c)
        self.assertIn((0,"николай павлович"), c)

    def test_gender_constraint_accepts_common_gender_surname(self):
        c = self.extract(["Фердыщенко вошёл. Фердыщенко заговорил. Он спросил господина Фердыщенка."])
        self.assertEqual(3, c[(0,"фердыщенко")].count)

    def test_preposition_distinguishes_family_from_male_instrumental(self):
        c = self.extract(["Генерал Епанчин ушёл. Он говорил с Епанчиным. Он отправился к Епанчиным."])
        self.assertEqual(2, c[(0,"епанчин")].count)
        self.assertEqual("family", c[(0,"семья епанчин")].kind)

    def test_unqualified_female_case_is_not_a_male_surname(self):
        c = self.extract(["Генерал Епанчин ушёл. " * 12 + "Генеральша Епанчина ответила. Он говорил о старшей Епанчиной."])
        self.assertNotIn("Епанчиной", c[(0,"епанчин")].forms)
        self.assertIn("Епанчиной", c[(0,"епанчина")].forms)


class ResponseTests(unittest.TestCase):
    def test_failed_atomic_write_preserves_previous_file(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "book.mytts-book"
            target.write_text('{"previous":true}', encoding="utf-8")
            with patch.object(b.json, "dump", side_effect=OSError("disk failure")):
                with self.assertRaises(OSError):
                    b.save(folder, target.name, {"new": True}, compact=True)
            self.assertEqual('{"previous":true}', target.read_text(encoding="utf-8"))
            self.assertEqual([target], list(Path(folder).iterdir()))

    def test_enum_thinking_and_boolean_thinking(self):
        self.assertEqual("high", b.thinking_control([False,"low","high","max"],True))
        self.assertEqual("medium", b.thinking_control([False,"low","medium","high"],True))
        self.assertIs(False,b.thinking_control([False,"low","high","max"],False))
        self.assertIs(True,b.thinking_control([False,True],True))
        self.assertIs(False,b.thinking_control([False],True))
        self.assertEqual("low",b.thinking_control(["low","high"],False))

    def test_duplicate_json_keys_rejected(self):
        self.assertIsNone(b.parse_answer('{"characters":[],"characters":[{}],"other":[]}'))

    def test_fenced_json_accepted(self):
        self.assertEqual({"characters": [], "other": []}, b.parse_answer('```json\n{"characters":[],"other":[]}\n```'))

    def test_wrong_schema_rejected(self):
        for data in [None, {}, {"characters": None, "other": []}, {"characters": [1], "other": []}]:
            self.assertFalse(b.valid_response(data))

    def test_verification_requires_identity(self):
        self.assertFalse(b.valid_response({"checks": [{"candidate":"c1","verdict":"same"}]}, True))

    def test_cache_identity_tracks_model_prompt_thinking(self):
        args = ["ollama", "https://ollama.com", "model", True, "prompt", 80000]
        original = b.cache_fingerprint(*args)
        for i,value in [(0,"deepseek"),(1,"https://other.example"),(2,"another"),(3,False),(4,"changed"),(5,1000)]:
            changed = args.copy(); changed[i]=value
            self.assertNotEqual(original, b.cache_fingerprint(*changed))


class ResumeTests(unittest.TestCase):
    def setUp(self):
        b._EFFORT_HINT.clear()
        self.addCleanup(b._EFFORT_HINT.clear)
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)
        self.candidates = {"c1": candidate("николай"), "c2": candidate("николай петрович")}
        self.r = dict(name="book", title="Fixture", sections=["s1"], candidates=list(self.candidates), prompt="Return JSON")
        b.save(str(self.folder), "candidates.json", {"candidates": [dict(c,id=k) for k,c in self.candidates.items()]})
        b.save(str(self.folder), "llm_request.json", {"scope":"book", "requests":[self.r]})
        self.answer = {"characters":[character("n",["c1","c2"])],"other":[]}
        self.checks = {"checks":[dict(character="n", anchor="c2", candidate="c1", verdict="same")]}
        self.args = Namespace(dir=str(self.folder), provider="ollama", model="test-model", endpoint="https://ollama.com", think=True, max_tokens=None, timeout=10, parallel=1, redo=False)
        self.calls = []
        def chat(provider, endpoint, model, key, prompt, think, limit, timeout):
            self.calls.append(prompt)
            reply = self.checks if prompt.startswith("Проверка") else self.answer
            return dict(content=json.dumps(reply), thinking_chars=1, done_reason="stop", prompt_tokens=1, output_tokens=1)
        self.chat = chat

    def tearDown(self):
        self.temp.cleanup()

    def run_llm(self, chat=None):
        with patch.dict(os.environ, {"OLLAMA_API_KEY":"not-a-real-key"}), patch.object(b, "load_env"), patch.object(b, "request_chat", side_effect=chat or self.chat), contextlib.redirect_stdout(io.StringIO()):
            b.llm(self.args)

    def test_votes_and_label_checks_are_cached_and_applied(self):
        self.candidates["c3"] = candidate("генерал", "title", source="Title", count=5)
        self.candidates["c3"]["contexts"] = ["[[генерал]] вошёл"] * 4
        self.r["candidates"] = list(self.candidates)
        b.save(str(self.folder), "candidates.json", {"book": "Fixture", "sections": [{"id": "s1", "title": "Test"}],
                                                      "candidates": [dict(c,id=k) for k,c in self.candidates.items()]})
        b.save(str(self.folder), "llm_request.json", {"scope":"book", "requests":[self.r]})
        self.answer = {"characters":[character("n",["c1","c2","c3"])],"other":[]}
        self.checks = {"checks":[dict(character="n", anchor="c2", candidate="c1", verdict="same"),
                                 dict(character="n", anchor="c2", candidate="c3", verdict="same")]}
        base = self.chat
        def chat(*args):
            if args[4].startswith("Кто назван"):
                self.calls.append(args[4])
                reply = {"answers": [{"n": n, "who": "n"} for n in range(1, 5)]}
                return dict(content=json.dumps(reply), thinking_chars=1, done_reason="stop", prompt_tokens=1, output_tokens=1)
            return base(*args)
        self.args.votes = 3
        self.run_llm(chat)
        self.assertEqual(5, len(self.calls))  # 3 основных, 1 по отрывкам, 1 проверка склеек
        self.calls.clear(); self.run_llm(chat)
        self.assertEqual([], self.calls)
        with contextlib.redirect_stdout(io.StringIO()):
            b.apply(Namespace(dir=str(self.folder), show=0, show_casts=0))
        cast = b.read_json(str(self.folder/"cast.json"))["casts"][0]
        self.assertEqual(["c2", "c1", "c3"], cast["characters"][0]["candidates"])

    def test_deepseek_provider_runs_whole_llm_stage(self):
        """Оригинальный API DeepSeek: /chat/completions, размышление, три ответа, отрывки, проверка склеек."""
        self.candidates["c3"] = candidate("генерал", "title", source="Title", count=5)
        self.candidates["c3"]["contexts"] = ["[[генерал]] вошёл"] * 3
        self.r["candidates"] = list(self.candidates)
        b.save(str(self.folder), "candidates.json", {"candidates": [dict(c,id=k) for k,c in self.candidates.items()]})
        b.save(str(self.folder), "llm_request.json", {"scope":"book", "requests":[self.r]})
        answer = {"characters":[character("n",["c1","c2","c3"])],"other":[]}
        checks = {"checks":[dict(character="n", anchor="c2", candidate="c1", verdict="same"),
                            dict(character="n", anchor="c2", candidate="c3", verdict="same")]}
        sent = []

        class Response(io.BytesIO):
            def __enter__(self): return self
            def __exit__(self, *a): return False

        def urlopen(req, timeout=None):
            body = json.loads(req.data)
            sent.append((req.full_url, body, req.get_header("Authorization")))
            prompt = body["messages"][0]["content"]
            if prompt.startswith("Проверка"):
                reply = checks
            elif prompt.startswith("Кто назван"):
                reply = {"answers": [{"n": n, "who": "n"} for n in range(1, 4)]}
            else:
                reply = answer
            data = {"choices": [{"message": {"content": json.dumps(reply), "reasoning_content": "…"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1}}
            return Response(json.dumps(data).encode())

        self.args.provider, self.args.model, self.args.endpoint, self.args.votes = "deepseek", None, None, 3
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "not-a-real-key"}), patch.object(b, "load_env"), \
                patch.object(b.urllib.request, "urlopen", side_effect=urlopen), contextlib.redirect_stdout(io.StringIO()):
            b.llm(self.args)
        self.assertEqual(5, len(sent))
        for url, body, auth in sent:
            self.assertEqual("https://api.deepseek.com/chat/completions", url)
            self.assertEqual("deepseek-flash", body["model"])
            self.assertEqual({"type": "enabled"}, body["thinking"])
            self.assertEqual("high", body["reasoning_effort"])
            self.assertEqual(0.6, body["temperature"])
            self.assertEqual("Bearer not-a-real-key", auth)

    def test_deepseek_token_limit_error_is_understood(self):
        error = b.urllib.error.HTTPError("https://api.deepseek.com", 400, "bad", {}, io.BytesIO(json.dumps(
            {"error": {"message": "Invalid max_tokens value, the valid range of max_tokens is [1, 65536]"}}).encode()))
        with patch.object(b.urllib.request, "urlopen", side_effect=error):
            with self.assertRaises(b.LLMError) as caught:
                b.request_chat("deepseek", "https://api.deepseek.com", "deepseek-flash", "k", "p", True, 80000, 10)
        self.assertEqual(65536, caught.exception.max_output_tokens)

    def test_length_cut_retries_with_shorter_thinking(self):
        b._EFFORT_HINT.clear()
        efforts = []
        def chat(*args, effort=None):
            efforts.append(effort)
            if len(efforts) == 1:
                return dict(content="", thinking_chars=9, thinking_control="high", done_reason="length", prompt_tokens=1, output_tokens=80000)
            return self.chat(*args)
        self.run_llm(chat)
        self.assertEqual([None, "medium"], efforts[:2])
        self.assertTrue((self.folder/"answers/book.failed1.txt").exists())
        b._EFFORT_HINT.clear()

    def test_minor_answer_deviations_are_normalized(self):
        answer = b.normalize_answer({"characters": [{"id": 7, "name": None, "gender": "Ж", "candidates": ["c1"]}], "other": []})
        self.assertTrue(b.valid_response(answer))
        self.assertEqual("f", answer["characters"][0]["gender"])

    def test_one_failed_vote_still_gives_majority_of_two(self):
        calls = []
        def chat(*args, effort=None):
            calls.append(effort)
            if args[4] == "Return JSON" and len(calls) in (2, 3, 4):  # второй ответ трижды обрывается
                return dict(content="", thinking_chars=9, thinking_control="high", done_reason="length", prompt_tokens=1, output_tokens=80000)
            return self.chat(*args)
        self.args.votes = 3
        b._EFFORT_HINT.clear()
        self.run_llm(chat)
        meta = b.load_answer(str(self.folder/"answers/book.meta.json"))
        self.assertEqual(2, meta["votes"])
        self.assertEqual("low", calls[4])  # следующие прогоны сразу с коротким размышлением
        b._EFFORT_HINT.clear()

    def test_resume_missing_verification_only(self):
        self.run_llm(); self.assertEqual(2,len(self.calls))
        (self.folder/"answers/book.verify.json").unlink()
        self.calls.clear(); self.run_llm()
        self.assertEqual(1,len(self.calls)); self.assertTrue(self.calls[0].startswith("Проверка"))

    def test_complete_resume_no_requests(self):
        self.run_llm(); self.calls.clear(); self.run_llm()
        self.assertEqual([],self.calls)

    def test_model_change_invalidates_cache(self):
        self.run_llm(); self.calls.clear(); self.args.model="another"
        self.run_llm(); self.assertEqual(2,len(self.calls))

    def test_prompt_change_invalidates_cache(self):
        self.run_llm(); self.calls.clear(); self.r["prompt"]="changed"
        b.save(str(self.folder),"llm_request.json",{"scope":"book","requests":[self.r]})
        self.run_llm(); self.assertEqual(2,len(self.calls)); self.assertEqual("changed", self.calls[0])

    def test_incomplete_verification_stops_pipeline(self):
        self.checks = {"checks":[]}
        with self.assertRaises(RuntimeError): self.run_llm()
        self.assertFalse((self.folder/"answers/book.verify.meta.json").exists())

    def test_truncated_json_is_not_cached(self):
        def truncated(*args):
            return dict(content=json.dumps(self.answer), thinking_chars=1, done_reason="length",prompt_tokens=1,output_tokens=1)
        with self.assertRaises(RuntimeError): self.run_llm(truncated)
        self.assertFalse((self.folder/"answers/book.json").exists())

    def test_apply_rejects_answers_for_old_extraction(self):
        self.run_llm()
        data = b.read_json(str(self.folder/"candidates.json"))
        data["book"] = "Fixture"
        data["sections"] = [{"id":"s1", "title":"Test"}]
        b.save(str(self.folder), "candidates.json", data)
        with self.assertRaises(ValueError):
            b.apply(Namespace(dir=str(self.folder),show=0,show_casts=0))

    def test_metadata_checks_input_identity(self):
        self.run_llm()
        meta=b.load_answer(str(self.folder/"answers/book.meta.json"))
        self.assertTrue(b.metadata_matches(meta,self.r["prompt"],str(self.folder)))
        self.assertFalse(b.metadata_matches(meta,"changed",str(self.folder)))
        self.assertFalse(b.metadata_matches({},self.r["prompt"],str(self.folder)))

    def test_transient_network_error_is_retried_sequentially(self):
        n = [0]
        def transient(*args):
            n[0] += 1
            if n[0] == 1:raise ConnectionResetError("reset")
            return self.chat(*args)
        with patch.object(b.time,"sleep"):
            self.run_llm(transient)
        self.assertEqual(3,n[0])
        self.assertEqual(2,len(self.calls))

    def test_server_token_limit_is_honored_and_cached(self):
        count=[0]
        def limited(*args):
            count[0] += 1
            if args[6] > 65536:raise b.LLMError(400,"cap",max_output_tokens=65536)
            return self.chat(*args)
        b._MODEL_LIMITS.clear()
        self.run_llm(limited)
        self.assertEqual(3,count[0])
        meta=b.load_answer(str(self.folder/"answers/book.meta.json"))
        self.assertEqual(65536,meta["effective_max_tokens"])
        self.assertEqual(80000,meta["max_tokens"])
        self.calls.clear();self.run_llm(limited);self.assertEqual([],self.calls)
        b._MODEL_LIMITS.clear()

    def test_api_failure_stops_pipeline(self):
        def failed(*args): raise b.LLMError(401,"denied")
        with self.assertRaises(RuntimeError): self.run_llm(failed)


class Fb2Tests(unittest.TestCase):
    XML = ('<?xml version="1.0" encoding="windows-1251"?><FictionBook xmlns="http://www.gribuser.ru/xml/fictionbook/2.0">'
           '<description><title-info><author><first-name>Иван</first-name><last-name>Петров</last-name></author>'
           '<book-title>Рассказы</book-title></title-info></description>'
           '<body><title><p>Рассказы</p></title>'
           '<section><title><p>Первый</p></title><p>Зина пришла домой.</p><section><p>Зина ушла.</p></section></section>'
           '<section><title><p>Второй</p></title><poem><stanza><v>Павлуша пел.</v></stanza></poem></section></body>'
           '<body name="notes"><section><p>Сноска про Ванду.</p></section></body></FictionBook>')

    def check(self, book):
        self.assertEqual("Рассказы", book.title)
        self.assertEqual("Иван Петров", book.author)
        self.assertEqual(["Первый", "Второй"], [s["title"] for s in book.sections])
        texts = [t for _, t in book.paragraphs]
        self.assertIn("Зина ушла.", texts)
        self.assertIn("Павлуша пел.", texts)
        self.assertNotIn("Сноска про Ванду.", texts)
        self.assertEqual(1, dict((t, s) for s, t in book.paragraphs)["Павлуша пел."])

    def test_fb2_in_windows_1251(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "book.fb2")
            Path(path).write_bytes(self.XML.encode("cp1251"))
            self.check(b.read_book(path))

    def test_zipped_fb2(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "book.fb2.zip")
            import zipfile
            with zipfile.ZipFile(path, "w") as z:
                z.writestr("book.fb2", self.XML.encode("cp1251"))
            self.check(b.read_book(path))


class ExportTests(unittest.TestCase):
    def test_rejected_title_does_not_erase_protagonist_voice_priority(self):
        with tempfile.TemporaryDirectory() as folder:
            candidates = {"c1": candidate("иван петрович", count=30, titles={"князь":1}),
                          "c2": candidate("николай павлович", count=100, titles={"князь":1}),
                          "c3": candidate("князь", kind="title", count=2000)}
            candidates["c2"]["speaker"] = 50
            candidates["c1"]["together_counts"] = {"c2":10}
            candidates["c2"]["together_counts"] = {"c1":10}
            request = dict(candidates=list(candidates), sections=["s1"], title="Fixture")
            answer = dict(characters=[character("main",["c1","c3"]),character("secondary",["c2"])],other=[])
            group = b.build_cast(request,answer,{("main","c1","c3"):"same"},candidates,"")
            self.assertEqual(["c1"],group["characters"][0]["candidates"])
            b.save(folder,"candidates.json",{"candidates":[dict(c,id=k) for k,c in candidates.items()]})
            b.save(folder,"llm_request.json",{})
            b.save(folder,"cast.json",{"book":"Fixture","input_sha256":b.artifact_identity(folder),"casts":[group]})
            b.save(folder,"voices.json",{"voices":[{"id":"unique","gender":"m"}],
                   "narrator":"author","other_m":"other_m","other_f":"other_f"})
            with contextlib.redirect_stdout(io.StringIO()):
                b.voices(Namespace(dir=folder,voices=str(Path(folder)/"voices.json"),min_speaker=2,
                                   min_mentions=30,max_shared=2,show_casts=0,show=0))
            chars={c["id"]:c for c in b.read_json(str(Path(folder)/"cast.json"))["casts"][0]["characters"]}
            self.assertEqual("own",chars["main"]["role"])
            self.assertEqual("other",chars["secondary"]["role"])

    def test_ambiguous_exported_forms_go_to_other(self):
        with tempfile.TemporaryDirectory() as folder:
            b.save(folder,"candidates.json",{"candidates":[dict(candidate("неизвестный",forms={"Николая":1}),id="c1")]})
            b.save(folder,"llm_request.json",{})
            b.save(folder,"book_index.json",{"book":{"title":"Fixture"},"sentences":{},"fingerprint":{}})
            b.save(folder,"cast.json",{"scope":"book","sections":[{"id":"s1","title":"Test","cast":0}],"input_sha256":b.artifact_identity(folder),"casts":[{"sections":["s1"],"characters":[{"id":"n","name":"Николай Петрович","gender":"m","speaker":3,"mentions":10,"forms":["Николая","Николай Петрович"],"role":"own"}],"other":[{"candidate":"c1","display":"Николая"}],"ambiguous":{"николая":["n","other"]}}]})
            with contextlib.redirect_stdout(io.StringIO()):
                b.export(Namespace(dir=folder,output=str(Path(folder)/"result.json"),max_forms=24,keep_other=False))
            cast=b.read_json(str(Path(folder)/"result.json"))["casts"][0]
            self.assertNotIn("Николая",cast["characters"][0]["forms"])
            self.assertIn("Николая",cast["other"])

    def test_empty_voice_pool_falls_back_to_other(self):
        with tempfile.TemporaryDirectory() as folder:
            b.save(folder,"candidates.json",{"candidates":[]})
            b.save(folder,"llm_request.json",{})
            b.save(folder,"voices.json",{"voices":[],"narrator":"author_voice","other_m":"male_voice","other_f":"female_voice"})
            b.save(folder,"cast.json",{"book":"Fixture","input_sha256":b.artifact_identity(folder),"casts":[{"sections":["s1"],"characters":[{"id":"n","name":"Name","gender":"m","speaker":4,"mentions":30}]}]})
            with contextlib.redirect_stdout(io.StringIO()):
                b.voices(Namespace(dir=folder,voices=str(Path(folder)/"voices.json"),min_speaker=2,min_mentions=30,max_shared=2,show_casts=0,show=0))
            ch=b.read_json(str(Path(folder)/"cast.json"))["casts"][0]["characters"][0]
            self.assertEqual("other",ch["role"])
            self.assertEqual("male_voice",ch["voice"])

    def test_export_rejects_stale_cast(self):
        with tempfile.TemporaryDirectory() as folder:
            b.save(folder,"cast.json",{"input_sha256":"old"})
            with self.assertRaises(ValueError):b.export(Namespace(dir=folder,output=None,max_forms=24,keep_other=False))


if __name__ == "__main__":
    unittest.main()
