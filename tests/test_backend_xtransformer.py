"""Unit tests for the XTransformer backend in Annif"""

import os.path as osp
from os import mknod
from unittest.mock import MagicMock, patch

import pytest
from scipy.sparse import csr_matrix, load_npz
import scipy.sparse as sp

import annif.backend
import annif.corpus
from annif.exception import NotInitializedException, NotSupportedException

pytest.importorskip("annif.backend.xtransformer")
XTransformer = annif.backend.xtransformer.XTransformer


@pytest.fixture
def mocked_xtransformer(datadir, project):
    model_mock = MagicMock()
    model_mock.save.side_effect = lambda x: mknod(osp.join(x, "test"))

    return patch.object(
        annif.backend.xtransformer.XTransformer, "train", return_value=model_mock
    )


def test_xtransformer_default_params(project):
    backend_type = annif.backend.get_backend("xtransformer")
    xtransformer = backend_type(
        backend_id="xtransfomer", config_params={}, project=project
    )
    expected = {
        "min_df": 1,
        "ngram": 1,
        "fix_clustering": False,
        "nr_splits": 16,
        "min_codes": None,
        "max_leaf_size": 100,
        "Cn": 0.5,
        "Cp": 5.0,
        "cost_sensitive_ranker": True,
        "rel_mode": "induce",
        "rel_norm": "l1",
        "neg_mining_chain": "tfn+man",
        "imbalanced_ratio": 0.0,
        "imbalanced_depth": 100,
        "max_match_clusters": 32768,
        "do_fine_tune": True,
        "model_shortcut": "distilbert-base-multilingual-uncased",
        "beam_size": 20,
        "limit": 100,
        "post_processor": "sigmoid",
        "negative_sampling": "tfn",
        "ensemble_method": "transformer-only",
        "threshold": 0.1,
        "loss_function": "squared-hinge",
        "truncate_length": 128,
        "hidden_droput_prob": 0.1,
        "batch_size": 32,
        "gradient_accumulation_steps": 1,
        "learning_rate": 1e-4,
        "weight_decay": 0.0,
        "adam_epsilon": 1e-8,
        "num_train_epochs": 1,
        "max_steps": 0,
        "lr_schedule": "linear",
        "warmup_steps": 0,
        "logging_steps": 100,
        "save_steps": 1000,
        "max_active_matching_labels": None,
        "max_num_labels_in_gpu": 65536,
        "use_gpu": True,
        "bootstrap_model": "linear"
    }
    actual = xtransformer.params
    assert len(actual) == len(expected)
    for param, val in expected.items():
        assert param in actual and actual[param] == val


def test_xtransformer_suggest_no_vectorizer(project):
    backend_type = annif.backend.get_backend("xtransformer")
    xtransformer = backend_type(
        backend_id="xtransfomer", config_params={}, project=project
    )
    with pytest.raises(NotInitializedException):
        xtransformer.suggest("example text")


def test_xtransformer_create_train_files(tmpdir, project, datadir):
    tmpfile = tmpdir.join("document.tsv")
    tmpfile.write(
        "nonexistent\thttp://example.com/nonexistent\n"
        + "arkeologia\thttp://www.yso.fi/onto/yso/p1265\n"
        + "...\thttp://example.com/none"
    )
    corpus = annif.corpus.DocumentFileTSV(str(tmpfile), project.subjects)
    backend_type = annif.backend.get_backend("xtransformer")
    xtransformer = backend_type(
        backend_id="xtransformer", config_params={}, project=project
    )
    input = (doc.text for doc in corpus.documents)
    veccorpus = xtransformer.create_vectorizer(input, {})
    xtransformer._create_train_files(veccorpus, corpus)
    assert datadir.join("xtransformer-train-X.npz").exists()
    assert datadir.join("xtransformer-train-y.npz").exists()
    assert datadir.join("xtransformer-train-raw.txt").exists()
    traindata = datadir.join("xtransformer-train-raw.txt").read().splitlines()
    assert len(traindata) == 1
    train_features = load_npz(str(datadir.join("xtransformer-train-X.npz")))
    assert train_features.shape[0] == 1
    train_labels = load_npz(str(datadir.join("xtransformer-train-y.npz")))
    assert train_labels.shape[0] == 1


def test_xtransformer_train(datadir, document_corpus, project, mocked_xtransformer):
    backend_type = annif.backend.get_backend("xtransformer")
    xtransformer = backend_type(
        backend_id="xtransfomer", config_params={}, project=project
    )

    with mocked_xtransformer as train_mock:
        xtransformer.train(document_corpus)

        train_mock.assert_called_once()
        first_arg = train_mock.call_args.args[0]
        kwargs = train_mock.call_args.kwargs
        assert len(first_arg.X_text) == 6402
        assert first_arg.X_feat.shape == (6402, 19507)
        assert first_arg.Y.shape == (6402, 130)
        expected_pred_params = XTransformer.PredParams.from_dict(
            {
                "beam_size": 20,
                "only_topk": 100,
                "post_processor": "sigmoid",
                "truncate_length": 128,
            },
            recursive=True,
        ).to_dict()

        expected_train_params = XTransformer.TrainParams.from_dict(
            {
                "do_fine_tune": True,
                "only_encoder": False,
                "fix_clustering": False,
                "max_match_clusters": 32768,
                "nr_splits": 16,
                "max_leaf_size": 100,
                "Cn": 0.5,
                "Cp": 5.0,
                "cost_sensitive_ranker": True,
                "rel_mode": "induce",
                "rel_norm": "l1",
                "neg_mining_chain": "tfn+man",
                "imbalanced_ratio": 0.0,
                "imbalanced_depth": 100,
                "model_shortcut": "distilbert-base-multilingual-uncased",
                "post_processor": "sigmoid",
                "negative_sampling": "tfn",
                "ensemble_method": "transformer-only",
                "threshold": 0.1,
                "loss_function": "squared-hinge",
                "truncate_length": 128,
                "hidden_droput_prob": 0.1,
                "batch_size": 32,
                "gradient_accumulation_steps": 1,
                "learning_rate": 1e-4,
                "weight_decay": 0.0,
                "adam_epsilon": 1e-8,
                "num_train_epochs": 1,
                "max_steps": 0,
                "lr_schedule": "linear",
                "warmup_steps": 0,
                "logging_steps": 100,
                "save_steps": 1000,
                "max_active_matching_labels": None,
                "max_num_labels_in_gpu": 65536,
                "use_gpu": True,
                "bootstrap_model": "linear"
            },
            recursive=True,
        ).to_dict()

        assert kwargs == {
            "clustering": None,
            "val_prob": None,
            "steps_scale": None,
            "label_feat": None,
            "beam_size": 20,
            "pred_params": expected_pred_params,
            "train_params": expected_train_params,
        }
        xtransformer._model.save.assert_called_once()
        assert datadir.join("xtransformer-model").check()


def test_xtransformer_train_cached(mocked_xtransformer, datadir, project):
    backend_type = annif.backend.get_backend("xtransformer")
    xtransformer = backend_type(
        backend_id="xtransfomer", config_params={}, project=project
    )
    xtransformer._create_train_files = MagicMock()
    xtransformer._create_model = MagicMock()
    with mocked_xtransformer:
        xtransformer.train("cached")
        xtransformer._create_train_files.assert_not_called()
        xtransformer._create_model.assert_called_once()


def test_xtransfomer_train_no_documents(datadir, project, empty_corpus):
    backend_type = annif.backend.get_backend("xtransformer")
    xtransformer = backend_type(
        backend_id="xtransfomer", config_params={}, project=project
    )
    with pytest.raises(NotSupportedException):
        xtransformer.train(empty_corpus)


def test_xtransformer_suggest(project):
    backend_type = annif.backend.get_backend("xtransformer")
    xtransformer = backend_type(
        backend_id="xtransfomer", config_params={}, project=project
    )
    xtransformer._model = MagicMock()
    xtransformer._model.predict.return_value = csr_matrix([0, 0.2, 0, 0, 0, 0.5, 0])
    results = xtransformer.suggest(
        [
            """Arkeologiaa sanotaan joskus myös
        muinaistutkimukseksi tai muinaistieteeksi. Se on humanistinen tiede
        tai oikeammin joukko tieteitä, jotka tutkivat ihmisen menneisyyttä.
        Tutkimusta tehdään analysoimalla muinaisjäännöksiä eli niitä jälkiä,
        joita ihmisten toiminta on jättänyt maaperään tai vesistöjen
        pohjaan."""
        ]
    )[0]
    xtransformer._model.predict.assert_called_once()

    ship_finds = project.subjects.by_uri("http://www.yso.fi/onto/yso/p8869")
    assert ship_finds in [result.subject_id for result in results]


def test_xtransformer_suggest_no_input(project, datadir):
    backend_type = annif.backend.get_backend("xtransformer")
    xtransformer = backend_type(
        backend_id="xtransfomer", config_params={"limit": 5}, project=project
    )
    xtransformer._model = MagicMock()
    results = xtransformer.suggest(["j"])
    assert len(results) == 0


def test_xtransformer_suggest_no_model(datadir, project):
    backend_type = annif.backend.get_backend("xtransformer")
    xtransformer = backend_type(
        backend_id="xtransfomer", config_params={}, project=project
    )
    datadir.remove()
    with pytest.raises(NotInitializedException):
        xtransformer.suggest("example text")


# ---------------- Vectorizer-only tests (PecosTfidfVectorizerMixin via XTransformer) ----------------

def _make_backend(project):
    backend_type = annif.backend.get_backend("xtransformer")
    return backend_type(backend_id="xtransformer", config_params={}, project=project)


def test_vectorizer_dict_defaults(project):
    backend = _make_backend(project)
    cfg = backend.vectorizer_dict(params={})
    assert cfg["type"] == "tfidf"
    kwargs = cfg["kwargs"]
    assert "base_vect_configs" in kwargs and isinstance(kwargs["base_vect_configs"], list)
    base = kwargs["base_vect_configs"][0]
    assert base["ngram_range"] == [1, 1]
    assert base["max_df_ratio"] == 0.98
    assert base["analyzer"] == "word"
    assert base["min_df_cnt"] == 1


def test_vectorizer_dict_overrides(project):
    backend = _make_backend(project)
    cfg = backend.vectorizer_dict(params={"ngram_range": [1, 2], "min_df": 3})
    base = cfg["kwargs"]["base_vect_configs"][0]
    assert base["ngram_range"] == [1, 2]
    assert base["min_df_cnt"] == 3


def test_create_vectorizer_trains_and_saves(datadir, project):
    backend = _make_backend(project)
    data = [
            """Arkeologiaa sanotaan joskus myös
        muinaistutkimukseksi tai muinaistieteeksi. Se on humanistinen tiede
        tai oikeammin joukko tieteitä, jotka tutkivat ihmisen menneisyyttä.
        Tutkimusta tehdään analysoimalla muinaisjäännöksiä eli niitä jälkiä,
        joita ihmisten toiminta on jättänyt maaperään tai vesistöjen
        pohjaan."""
        ]
    mat = backend.create_vectorizer(data, params={"ngram_range": [1, 1], "min_df": 1})
    assert backend.vectorizer is not None
    assert sp.issparse(mat)
    assert mat.shape[0] == len(data)
    vec_path = osp.join(str(datadir), backend.VECTORIZER_FILE)
    assert osp.exists(vec_path)


def test_initialize_vectorizer_loads_existing(datadir, project):
    backend = _make_backend(project)
    data = ["alpha", "beta", "gamma"]
    backend.create_vectorizer(data, params={})
    backend.vectorizer = None
    backend.initialize_vectorizer()
    assert backend.vectorizer is not None
