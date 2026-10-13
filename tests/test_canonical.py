import unittest

import mytts_book as b


class CanonicalTests(unittest.TestCase):
    """The same spelling as BookTextSpacing.canonical in MyTTS (BookFingerprintTest checks the Kotlin side)."""

    def test_invisible_and_composed(self):
        self.assertEqual("молоко", b.canonical("мо­локо"))
        self.assertEqual("Он пришёл", b.canonical("﻿Он пришёл"))
        self.assertEqual("ёж и йод", b.canonical("ёж и йод"))
        self.assertEqual("молоко́", b.canonical("молоко̀"))

    def test_latin_inside_russian_words_only(self):
        self.assertEqual("молоко́ и Хорошо", b.canonical("молокó и Xорошо"))
        self.assertEqual("iPhone-ом и Shakespeare", b.canonical("iPhone-ом и Shakespeare"))

    def test_fingerprints_ignore_the_typing(self):
        plain = "Князь Мышкин вошёл в комнату и тихо сел у окна."
        typed = "Кня­зь Мышкин вошёл в кoмнату и тихо сел у окна."
        self.assertEqual(b.sentence_fingerprints(plain), b.sentence_fingerprints(typed))
        self.assertEqual(b.letters(plain), b.letters(typed))


if __name__ == "__main__":
    unittest.main()
