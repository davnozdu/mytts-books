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


class ExtractorTests(unittest.TestCase):
    def extract(self, paragraphs):
        return b.Extractor().run(b.Book("Fixture", "", [{"id": "s1", "title": "Test"}], [(0,t) for t in paragraphs]), False)

    def test_epanchin_spouses_stay_separate(self):
        c = self.extract(["Генерал Епанчин вошёл. Генеральша Епанчина ответила. Генерал Епанчин ушёл."])
        self.assertIn((0, "епанчин"), c)
        self.assertIn((0, "епанчина"), c)
        self.assertEqual("m", b.gender_of(c[(0, "епанчин")]))
        self.assertEqual("f", b.gender_of(c[(0, "епанчина")]))

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


class ExportTests(unittest.TestCase):
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
