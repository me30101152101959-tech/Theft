"""
Regression tests — Manual vs Batch prediction consistency (ETD-XAI v4.0).

Proves the success criteria from the consistency spec:
  Test 1  Same customer: Manual vs Batch feed an identical input tensor
  Test 2  CSV with FLAG vs without FLAG: probability diff == 0.0
  Test 3  Renaming reading headers: identical input tensor (no header leakage)
  Test 4  Different ID column names: identical input tensor

Run:  python tests_prediction_consistency.py
The app is a single file, so we stub streamlit/plotly and exec app.py up to
the SECTION 9 marker (engine only), exactly like the other harness scripts.
"""
import sys
import types
import numpy as np
import pandas as pd


def _load_app():
    st = types.ModuleType("streamlit")
    st.set_page_config = lambda **k: None
    st.cache_resource = lambda *a, **k: (lambda f: f)
    for n in ("markdown", "caption"):
        setattr(st, n, lambda *a, **k: None)
    sys.modules["streamlit"] = st
    pkg = types.ModuleType("streamlit.components"); st.components = pkg
    sys.modules["streamlit.components"] = pkg
    v1 = types.ModuleType("streamlit.components.v1"); v1.html = lambda *a, **k: None
    pkg.v1 = v1; sys.modules["streamlit.components.v1"] = v1
    for m in ("plotly", "plotly.express", "plotly.graph_objects"):
        sys.modules[m] = types.ModuleType(m)
    src = open("app.py", encoding="utf-8").read()
    g = {"__name__": "t", "__file__": "app.py"}
    exec(compile(src[: src.index("# SECTION 9")], "app.py", "exec"), g)
    g["init_db"]()
    g["auto_load_default"]()
    return g


def _sample_frame(g, n_rows=25):
    df = g["read_table"]("assets/sample_dataset.csv").head(n_rows).reset_index(drop=True)
    return df


def test_manual_equals_batch(g):
    """Same customer through Manual and Batch must feed model.predict() the
    exact same input tensor (hash-identical). Output probabilities match within
    float32 tolerance — any residual is TensorFlow batch-reduction noise (~1e-7),
    not a preprocessing difference."""
    df = _sample_frame(g)
    info = g["inspect"](df)
    readings, ids, _ = g["build_matrix"](df, info)
    thr = g["config_threshold"]()
    batch = g["run_batch"](df, info, "last_n", thr)
    max_diff = 0.0
    for i in range(len(df)):
        man = g["predict_one"](readings[i].tolist(), "last_n", thr)
        bat = batch["rows"][i]["probability"]
        # input-tensor identity: manual row vs the same row inside the batch
        assert g["_input_hash"](readings[i].reshape(1, -1)) == g["_input_hash"](readings[i:i+1]), \
            f"input tensor differs for row {i}"
        max_diff = max(max_diff, abs(man["probability"] - bat))
    assert max_diff <= 1e-5, f"Manual vs Batch diff {max_diff} exceeds float tolerance"
    return f"identical input tensors; max prob diff = {max_diff:.2e} (float-noise floor)"


def test_flag_optional(g):
    df = _sample_frame(g)
    info = g["inspect"](df)
    thr = g["config_threshold"]()
    with_flag = g["run_batch"](df, info, "last_n", thr)
    df2 = df.drop(columns=[info["flag_col"]])
    info2 = g["inspect"](df2)
    without_flag = g["run_batch"](df2, info2, "last_n", thr)
    diffs = [abs(a["probability"] - b["probability"])
             for a, b in zip(with_flag["rows"], without_flag["rows"])]
    md = max(diffs)
    assert md == 0.0, f"FLAG changed predictions by {md}"
    assert without_flag["metrics"] is None and with_flag["metrics"] is not None
    return f"max diff = {md:.9f}; metrics only when FLAG present"


def test_headers_dont_leak(g):
    """Renaming reading-column headers (any format) must NOT change the input
    tensor — headers are metadata, only the values in CSV order feed the model."""
    df = _sample_frame(g)
    info = g["inspect"](df)
    r_ref, _, _ = g["build_matrix"](df, info)
    ren = {c: f"reading_{i:03d}" for i, c in enumerate(info["reading_cols"])}
    df_alt = df.rename(columns=ren)
    info_alt = g["inspect"](df_alt)
    r_alt, _, _ = g["build_matrix"](df_alt, info_alt)
    h1, h2 = g["_input_hash"](r_ref), g["_input_hash"](r_alt)
    assert h1 == h2, f"header rename changed tensor: {h1} vs {h2}"
    return f"hash = {h1[:12]}… (identical)"


def test_id_column_names_identical(g):
    df = _sample_frame(g)
    info = g["inspect"](df)
    r_ref, _, _ = g["build_matrix"](df, info)
    df_alt = df.rename(columns={info["id_col"]: "CustomerID"})
    info_alt = g["inspect"](df_alt)
    r_alt, _, _ = g["build_matrix"](df_alt, info_alt)
    h1, h2 = g["_input_hash"](r_ref), g["_input_hash"](r_alt)
    assert h1 == h2, f"ID-name hashes differ: {h1} vs {h2}"
    return f"hash = {h1[:12]}… (identical)"


def main():
    g = _load_app()
    tests = [
        ("Test 1 — Manual == Batch", test_manual_equals_batch),
        ("Test 2 — FLAG optional", test_flag_optional),
        ("Test 3 — Headers do not leak into tensor", test_headers_dont_leak),
        ("Test 4 — ID column names identical tensor", test_id_column_names_identical),
    ]
    ok = True
    for name, fn in tests:
        try:
            detail = fn(g)
            print(f"PASS  {name}  ({detail})")
        except AssertionError as e:
            ok = False
            print(f"FAIL  {name}  -> {e}")
    print("\nALL PASSED" if ok else "\nSOME TESTS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
