"""Unit tests for jieba analyzer in Annif"""

import pickle

import pytest

import annif.analyzer
import annif.analyzer.jieba
from annif.exception import OperationFailedException

pytestmark = pytest.mark.skipif(
    not annif.analyzer.jieba.JiebaAnalyzer.is_available(),
    reason="rjieba is required",
)

# Custom dictionaries need Jieba.add_word, which was added in rjieba 0.2.2
needs_add_word = pytest.mark.skipif(
    not annif.analyzer.jieba.JiebaAnalyzer.is_available()
    or not hasattr(__import__("rjieba").Jieba(), "add_word"),
    reason="rjieba 0.2.2 or newer is required for custom dictionaries",
)

TEXT = "南海夏季风爆发前后大气环流的演变。利用NCEP再分析资料，研究了季风爆发！"


def test_jieba_tokenize_words():
    analyzer = annif.analyzer.get_analyzer("jieba")
    words = analyzer.tokenize_words(TEXT)
    assert words == [
        "南海",
        "夏季",
        "风",
        "爆发",
        "前后",
        "大气",
        "环流",
        "的",
        "演变",
        "利用",
        "ncep",
        "再",
        "分析",
        "资料",
        "研究",
        "了",
        "季风",
        "爆发",
    ]


def test_jieba_tokenize_words_no_filter():
    analyzer = annif.analyzer.get_analyzer("jieba")
    words = analyzer.tokenize_words("大气环流的演变。", filter=False)
    assert words == ["大气", "环流", "的", "演变", "。"]


def test_jieba_token_min_length():
    analyzer = annif.analyzer.get_analyzer("jieba(token_min_length=2)")
    words = analyzer.tokenize_words("大气环流的演变")
    assert words == ["大气", "环流", "演变"]


@needs_add_word
def test_jieba_userdict(tmp_path):
    userdict = tmp_path / "userdict.txt"
    userdict.write_text("夏季风\n大气环流\n", encoding="utf-8")
    analyzer = annif.analyzer.get_analyzer(f"jieba(userdict={userdict})")
    words = analyzer.tokenize_words(TEXT)
    assert "夏季风" in words
    assert "大气环流" in words
    assert "夏季" not in words


@needs_add_word
def test_jieba_userdict_positional(tmp_path):
    userdict = tmp_path / "userdict.txt"
    userdict.write_text("夏季风\n", encoding="utf-8")
    analyzer = annif.analyzer.get_analyzer(f"jieba({userdict})")
    assert "夏季风" in analyzer.tokenize_words(TEXT)


@needs_add_word
def test_jieba_userdict_freq_and_tag(tmp_path):
    userdict = tmp_path / "userdict.txt"
    userdict.write_text("夏季风 5 n\n\n大气环流 nz\n", encoding="utf-8")
    analyzer = annif.analyzer.get_analyzer(f"jieba(userdict={userdict})")
    words = analyzer.tokenize_words(TEXT)
    assert "夏季风" in words
    assert "大气环流" in words


@needs_add_word
@pytest.mark.parametrize(
    "text",
    ["B超检查和DNA甲基化", "Ｂ超检查和ＤＮＡ甲基化", "b超检查和dna甲基化"],
)
def test_jieba_userdict_normalized_entries(tmp_path, text):
    userdict = tmp_path / "userdict.txt"
    # a full-width entry and a mixed-case entry
    userdict.write_text("Ｂ超检查\nDNA甲基化\n", encoding="utf-8")
    analyzer = annif.analyzer.get_analyzer(f"jieba(userdict={userdict})")
    words = analyzer.tokenize_words(text)
    assert "b超检查" in words
    assert "dna甲基化" in words


def test_jieba_userdict_requires_add_word(tmp_path, monkeypatch):
    class OldJieba:  # mimics rjieba < 0.2.2, which has no add_word
        def cut(self, text, hmm=True):
            return [text]

    monkeypatch.setattr("rjieba.Jieba", OldJieba)
    userdict = tmp_path / "userdict.txt"
    userdict.write_text("夏季风\n", encoding="utf-8")
    analyzer = annif.analyzer.get_analyzer(f"jieba(userdict={userdict})")
    with pytest.raises(OperationFailedException):
        analyzer.tokenize_words(TEXT)


def test_jieba_hmm_disabled():
    analyzer = annif.analyzer.get_analyzer("jieba(hmm=0)")
    assert analyzer.hmm is False
    assert analyzer.tokenize_words("大气环流的演变") == ["大气", "环流", "的", "演变"]


def test_jieba_fullwidth_normalized():
    analyzer = annif.analyzer.get_analyzer("jieba")
    assert "ncep" in analyzer.tokenize_words("ＮＣＥＰ资料")


def test_jieba_tokenize_sentences():
    analyzer = annif.analyzer.get_analyzer("jieba")
    sentences = analyzer.tokenize_sentences(TEXT)
    assert sentences == [
        "南海夏季风爆发前后大气环流的演变",
        "利用NCEP再分析资料，研究了季风爆发",
    ]


@pytest.mark.parametrize(
    "text,expected",
    [
        ("第一句.第二句.", ["第一句", "第二句"]),
        ("增长了3.5倍.结论如下", ["增长了3.5倍", "结论如下"]),
        ("美国(U.S.)经济", ["美国(U.S.)经济"]),
        ("（见附录）.下一句", ["（见附录）", "下一句"]),
    ],
)
def test_jieba_tokenize_sentences_ascii_full_stop(text, expected):
    analyzer = annif.analyzer.get_analyzer("jieba")
    assert analyzer.tokenize_sentences(text) == expected


def test_jieba_pickle():
    analyzer = annif.analyzer.get_analyzer("jieba")
    words = analyzer.tokenize_words(TEXT)
    restored = pickle.loads(pickle.dumps(analyzer))
    assert restored.tokenize_words(TEXT) == words
