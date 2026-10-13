"""Unit tests for jieba analyzer in Annif"""

import pickle

import pytest

import annif.analyzer
import annif.analyzer.jieba

pytestmark = pytest.mark.skipif(
    not annif.analyzer.jieba.JiebaAnalyzer.is_available(),
    reason="jieba is required",
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


def test_jieba_userdict(tmp_path):
    userdict = tmp_path / "userdict.txt"
    userdict.write_text("夏季风\n大气环流\n", encoding="utf-8")
    analyzer = annif.analyzer.get_analyzer(f"jieba(userdict={userdict})")
    words = analyzer.tokenize_words(TEXT)
    assert "夏季风" in words
    assert "大气环流" in words
    assert "夏季" not in words


def test_jieba_userdict_positional(tmp_path):
    userdict = tmp_path / "userdict.txt"
    userdict.write_text("夏季风\n", encoding="utf-8")
    analyzer = annif.analyzer.get_analyzer(f"jieba({userdict})")
    assert "夏季风" in analyzer.tokenize_words(TEXT)


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


def test_jieba_pickle():
    analyzer = annif.analyzer.get_analyzer("jieba")
    words = analyzer.tokenize_words(TEXT)
    restored = pickle.loads(pickle.dumps(analyzer))
    assert restored.tokenize_words(TEXT) == words
