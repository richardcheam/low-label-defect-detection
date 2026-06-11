from simclr_hpl.tracking import ExperimentTracker


def test_enabled_tracker_logs_params_and_metrics(tmp_path):
    uri = (tmp_path / "mlruns").as_uri()
    tracker = ExperimentTracker(enabled=True, tracking_uri=uri, experiment="unit-test")
    with tracker.run(run_name="r1"):
        tracker.log_params({"lr": 0.001, "encoder": "resnet18"})
        tracker.log_metrics({"accuracy": 0.91, "loss": 0.2})

    import mlflow
    mlflow.set_tracking_uri(uri)
    runs = mlflow.search_runs(experiment_names=["unit-test"])
    assert len(runs) == 1
    assert runs.iloc[0]["params.encoder"] == "resnet18"
    assert float(runs.iloc[0]["metrics.accuracy"]) == 0.91


def test_disabled_tracker_is_noop(tmp_path):
    # Disabled tracker must not raise and must not create any runs.
    tracker = ExperimentTracker(enabled=False)
    with tracker.run(run_name="r"):
        tracker.log_params({"a": 1})
        tracker.log_metrics({"m": 1.0})
    # nothing to assert beyond "no exception"; the context+methods are safe no-ops
