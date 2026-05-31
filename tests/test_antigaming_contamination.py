"""Contamination scan + flag report (ARCHITECTURE.md §6.4)."""

from darwin.antigaming import AntiGamingFlag, AntiGamingReport, contamination_scan, word_ngrams


# ------------------------------------------------------------------ report model


def test_report_count_sums_severities():
    report = AntiGamingReport(
        flags=[
            AntiGamingFlag("contamination", "x"),
            AntiGamingFlag("genome_hack", "y", severity=2),
        ]
    )
    assert report.count == 3
    assert report.kinds == {"contamination", "genome_hack"}
    assert report.clean is False


def test_empty_report_is_clean():
    report = AntiGamingReport()
    assert report.clean is True
    assert report.count == 0


# ------------------------------------------------------------------ n-grams


def test_word_ngrams_basic():
    grams = word_ngrams("the quick brown fox", 2)
    assert ("the", "quick") in grams
    assert ("brown", "fox") in grams
    assert len(grams) == 3


def test_word_ngrams_short_text_is_single_gram():
    # fewer tokens than n -> the whole sequence is one gram so tiny eval answers still match
    assert word_ngrams("yes", 8) == {("yes",)}
    assert word_ngrams("", 8) == set()


# ------------------------------------------------------------------ scan (§6.4)


def test_clean_data_raises_no_flags():
    data = ["def load(): return datasets.load('clean-corpus')"]
    evals = ["What is the capital of France? Paris.", "Compute 2 + 2 = 4"]
    assert contamination_scan(data, evals, n=4) == []


def test_verbatim_eval_in_data_is_flagged():
    eval_item = "the integral of x squared dx equals x cubed over three plus c"
    data = [f"train_examples = [\n  '{eval_item}',\n]"]
    flags = contamination_scan(data, [eval_item, "an unrelated clean example here"], n=6)
    assert len(flags) == 1
    assert flags[0].kind == "contamination"
    assert "eval item 0" in flags[0].detail


def test_scan_caps_flag_count():
    leaked = "this exact phrase appears verbatim in the training data mixture"
    data = [leaked]
    evals = [leaked] * 50
    flags = contamination_scan(data, evals, n=6, max_flags=20)
    assert len(flags) == 20


def test_empty_data_never_flags():
    assert contamination_scan([], ["anything at all here"], n=4) == []
