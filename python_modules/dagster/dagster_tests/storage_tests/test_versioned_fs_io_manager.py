"""Tests for opt-in version retention and historical replay in the built-in
filesystem IO manager.

Covers the settlement-ETL scenario from the feature request:

* materializations are retained as immutable per-run versions rather than
  overwritten;
* a manifest records every materialization and the exact input versions each run
  read, including multi-partition loads;
* a replay run binds its inputs to a historical run; partitions it recomputes are
  served from its own new versions while everything else stays pinned to the old
  versions;
* a partial re-run after a failure still cannot drift back to latest data, and a
  missing bound version terminates the run explicitly;
* the default (versioning disabled) behavior is unchanged.
"""

import json
import os
import tempfile

import dagster as dg
import pytest
from dagster._core.definitions.assets.graph.asset_graph import AssetGraph
from dagster._core.storage.versioned_io_manager import (
    MANIFEST_FILENAME,
    VERSION_REPLAY_RUN_ID_TAG,
    VERSIONS_DIRNAME,
)

REPLAY_TAG = VERSION_REPLAY_RUN_ID_TAG


def _load_manifest(base_dir: str) -> dict:
    with open(os.path.join(base_dir, MANIFEST_FILENAME), encoding="utf-8") as f:
        return json.load(f)


def _input_bindings(manifest: dict, run_id: str) -> dict:
    return manifest["runs"][run_id]["inputs"]


def _ak(name: str) -> str:
    # Manifest keys use AssetKey.to_string().
    return dg.AssetKey([name]).to_string()


def _assert_replay_error(exc_info: pytest.ExceptionInfo) -> None:
    # Execution may surface the error directly or wrapped in a load-input error;
    # walk the cause chain and require the explicit replay error type.
    error: BaseException | None = exc_info.value
    seen = []
    while error is not None:
        if isinstance(error, dg.DagsterVersionReplayError):
            return
        seen.append(type(error).__name__)
        error = error.__cause__
    raise AssertionError(f"expected DagsterVersionReplayError in cause chain, got: {seen}")


def _version_file(base_dir: str, asset_path: list[str], run_id: str) -> str:
    # <base>/<asset parent>/.versions/<asset name>/<run id>
    head, tail = asset_path[:-1], asset_path[-1]
    return os.path.join(base_dir, *head, VERSIONS_DIRNAME, tail, run_id)


def test_disabled_by_default_is_backward_compatible():
    with tempfile.TemporaryDirectory() as tmpdir:
        io_manager = dg.FilesystemIOManager(base_dir=tmpdir)

        @dg.asset(io_manager_def=io_manager)
        def upstream():
            return 1

        @dg.asset(io_manager_def=io_manager)
        def downstream(upstream):
            return upstream + 1

        result = dg.materialize(
            [upstream, downstream],
            instance=dg.DagsterInstance.ephemeral(tempdir=tempfile.mkdtemp()),
        )
        assert result.success
        # No manifest, no .versions tree, canonical overwrite layout intact.
        assert not os.path.exists(os.path.join(tmpdir, MANIFEST_FILENAME))
        assert not os.path.exists(os.path.join(tmpdir, VERSIONS_DIRNAME))
        canonical = os.path.join(tmpdir, "upstream")
        assert os.path.isfile(canonical)


def test_versioned_materializations_are_retained_and_manifest_records_reads():
    with tempfile.TemporaryDirectory() as tmpdir:
        io_manager = dg.FilesystemIOManager(base_dir=tmpdir, versioned=True)

        @dg.asset(io_manager_def=io_manager)
        def upstream():
            return 1

        @dg.asset(io_manager_def=io_manager)
        def downstream(upstream):
            return upstream + 1

        defs = dg.Definitions(assets=[upstream, downstream])
        instance = dg.DagsterInstance.ephemeral(tempdir=tempfile.mkdtemp())

        first = defs.get_implicit_global_asset_job_def().execute_in_process(instance=instance)
        assert first.success
        first_run = first.run_id
        v1_upstream = _version_file(tmpdir, ["upstream"], first_run)
        v1_downstream = _version_file(tmpdir, ["downstream"], first_run)
        assert os.path.isfile(v1_upstream)
        assert os.path.isfile(v1_downstream)

        # Re-materialize both assets after the "upstream partition was rewritten".
        second = defs.get_implicit_global_asset_job_def().execute_in_process(instance=instance)
        assert second.success
        second_run = second.run_id
        assert os.path.isfile(_version_file(tmpdir, ["upstream"], second_run))
        assert os.path.isfile(_version_file(tmpdir, ["downstream"], second_run))
        # The old versions are retained: nothing was overwritten.
        assert os.path.isfile(v1_upstream)
        assert os.path.isfile(v1_downstream)

        manifest = _load_manifest(tmpdir)
        # The second run's downstream input is pinned to the second run's upstream,
        # never the first run's stale one.
        assert _input_bindings(manifest, second_run)[_ak("upstream")][""] == second_run
        assert _input_bindings(manifest, first_run)[_ak("upstream")][""] == first_run


def test_replay_reproduces_historical_inputs_after_upstream_rewrite():
    with tempfile.TemporaryDirectory() as tmpdir:
        io_manager = dg.FilesystemIOManager(base_dir=tmpdir, versioned=True)
        holder = {"value": 1}

        @dg.asset(io_manager_def=io_manager)
        def upstream():
            return holder["value"]

        @dg.asset(io_manager_def=io_manager)
        def report(upstream):
            return {"consumed": upstream}

        defs = dg.Definitions(assets=[upstream, report])
        instance = dg.DagsterInstance.ephemeral(tempdir=tempfile.mkdtemp())
        job = defs.get_implicit_global_asset_job_def()

        historical = job.execute_in_process(instance=instance)
        assert historical.success
        historical_run = historical.run_id
        assert historical.output_for_node("report") == {"consumed": 1}

        # Upstream is rewritten: a new materialization carries new data.
        holder["value"] = 2
        rewrite = job.execute_in_process(instance=instance)
        assert rewrite.success
        assert rewrite.output_for_node("upstream") == 2

        # Replay only the report, bound to the historical run. It must read the old
        # upstream version (1), not the current latest (2).
        replay = job.execute_in_process(
            instance=instance,
            tags={REPLAY_TAG: historical_run},
            asset_selection=[dg.AssetKey("report")],
        )
        assert replay.success
        assert replay.output_for_node("report") == {"consumed": 1}

        manifest = _load_manifest(tmpdir)
        replay_run = replay.run_id
        # The replay's input binding points at the historical producing run, and its
        # own output is a brand-new version rather than an overwrite.
        assert _input_bindings(manifest, replay_run)[_ak("upstream")][""] == historical_run
        assert replay_run != historical_run
        assert os.path.isfile(_version_file(tmpdir, ["report"], replay_run))
        assert os.path.isfile(_version_file(tmpdir, ["report"], historical_run))
        # The replay never materialized upstream itself, so no upstream version
        # exists under its run id.
        assert not os.path.exists(_version_file(tmpdir, ["upstream"], replay_run))


def test_partial_rerun_mixes_recomputed_and_bound_versions_without_drift():
    with tempfile.TemporaryDirectory() as tmpdir:
        io_manager = dg.FilesystemIOManager(base_dir=tmpdir, versioned=True)
        holder = {"value": 1}

        @dg.asset(io_manager_def=io_manager)
        def upstream():
            return holder["value"]

        @dg.asset(io_manager_def=io_manager)
        def report(upstream):
            return {"consumed": upstream}

        defs = dg.Definitions(assets=[upstream, report])
        instance = dg.DagsterInstance.ephemeral(tempdir=tempfile.mkdtemp())
        job = defs.get_implicit_global_asset_job_def()

        historical = job.execute_in_process(instance=instance)
        historical_run = historical.run_id

        # Upstream is rewritten on disk to a new value.
        holder["value"] = 2
        job.execute_in_process(instance=instance)

        # Failure recovery style: rerun BOTH assets but keep the historical binding.
        # The recomputed upstream (2) must feed the report; the report output is a new
        # version. No read may silently jump to "latest".
        rerun = job.execute_in_process(instance=instance, tags={REPLAY_TAG: historical_run})
        assert rerun.success
        assert rerun.output_for_node("upstream") == 2
        assert rerun.output_for_node("report") == {"consumed": 2}

        manifest = _load_manifest(tmpdir)
        rerun_run = rerun.run_id
        # Same-run recomputation wins over the historical binding.
        assert _input_bindings(manifest, rerun_run)[_ak("upstream")][""] == rerun_run


def test_replay_terminates_when_bound_version_is_missing():
    with tempfile.TemporaryDirectory() as tmpdir:
        io_manager = dg.FilesystemIOManager(base_dir=tmpdir, versioned=True)

        @dg.asset(io_manager_def=io_manager)
        def upstream():
            return 1

        @dg.asset(io_manager_def=io_manager)
        def report(upstream):
            return upstream

        defs = dg.Definitions(assets=[upstream, report])
        instance = dg.DagsterInstance.ephemeral(tempdir=tempfile.mkdtemp())
        job = defs.get_implicit_global_asset_job_def()

        historical = job.execute_in_process(instance=instance)
        historical_run = historical.run_id

        # Delete the retained upstream version the historical run consumed.
        bound_version = _version_file(tmpdir, ["upstream"], historical_run)
        os.remove(bound_version)

        with pytest.raises(Exception) as exc_info:
            job.execute_in_process(
                instance=instance,
                tags={REPLAY_TAG: historical_run},
                asset_selection=[dg.AssetKey("report")],
            )
        # Explicit termination type, never a silent fallback or a bare FileNotFound.
        _assert_replay_error(exc_info)


def test_replay_terminates_when_historical_run_unknown():
    with tempfile.TemporaryDirectory() as tmpdir:
        io_manager = dg.FilesystemIOManager(base_dir=tmpdir, versioned=True)

        @dg.asset(io_manager_def=io_manager)
        def upstream():
            return 1

        @dg.asset(io_manager_def=io_manager)
        def report(upstream):
            return upstream

        defs = dg.Definitions(assets=[upstream, report])
        instance = dg.DagsterInstance.ephemeral(tempdir=tempfile.mkdtemp())
        job = defs.get_implicit_global_asset_job_def()
        job.execute_in_process(instance=instance)

        with pytest.raises(Exception) as exc_info:
            job.execute_in_process(
                instance=instance,
                tags={REPLAY_TAG: "run-that-never-happened"},
                asset_selection=[dg.AssetKey("report")],
            )
        _assert_replay_error(exc_info)


def test_partitioned_replay_pins_each_input_partition():
    with tempfile.TemporaryDirectory() as tmpdir:
        io_manager = dg.FilesystemIOManager(base_dir=tmpdir, versioned=True)
        partitions_def = dg.StaticPartitionsDefinition(["a", "b"])
        holder = {"a": 1, "b": 1}

        @dg.asset(partitions_def=partitions_def, io_manager_def=io_manager)
        def upstream(context):
            return holder[context.partition_key]

        @dg.asset(partitions_def=partitions_def, io_manager_def=io_manager)
        def report(context, upstream):
            return {context.partition_key: upstream}

        defs = dg.Definitions(assets=[upstream, report])
        instance = dg.DagsterInstance.ephemeral(tempdir=tempfile.mkdtemp())
        job = defs.get_implicit_global_asset_job_def()

        historical_runs = {}
        for key in ("a", "b"):
            result = job.execute_in_process(instance=instance, partition_key=key)
            assert result.success
            historical_runs[key] = result.run_id

        # Rewrite partition "a" only; partition "b" keeps its old version.
        holder["a"] = 2
        rewrite = job.execute_in_process(instance=instance, partition_key="a")
        rewrite_run = rewrite.run_id

        # Replay the report for partition "a": must consume the historical version 1.
        replay_a = job.execute_in_process(
            instance=instance,
            partition_key="a",
            tags={REPLAY_TAG: historical_runs["a"]},
            asset_selection=[dg.AssetKey("report")],
        )
        assert replay_a.success
        # output_for_node nests partitioned outputs under their partition key.
        assert replay_a.output_for_node("report") == {"a": {"a": 1}}

        manifest = _load_manifest(tmpdir)
        replay_a_run = replay_a.run_id
        assert _input_bindings(manifest, replay_a_run)[_ak("upstream")]["a"] == historical_runs["a"]
        # The newer rewrite version still exists and is untouched.
        assert os.path.isfile(_version_file(tmpdir, ["upstream", "a"], rewrite_run))


def test_multipartition_summary_replay_binds_each_dimension_partition():
    with tempfile.TemporaryDirectory() as tmpdir:
        io_manager = dg.FilesystemIOManager(base_dir=tmpdir, versioned=True)
        multipart = dg.MultiPartitionsDefinition(
            {
                "dim": dg.StaticPartitionsDefinition(["x", "y"]),
                "date": dg.StaticPartitionsDefinition(["2024-01-01"]),
            }
        )
        holder = {("x", "2024-01-01"): 1, ("y", "2024-01-01"): 1}

        @dg.asset(partitions_def=multipart, io_manager_def=io_manager)
        def upstream(context):
            key = (
                context.partition_key.keys_by_dimension["dim"],
                context.partition_key.keys_by_dimension["date"],
            )
            return holder.get(key, 0)

        @dg.asset(
            io_manager_def=io_manager,
            ins={
                "upstream": dg.AssetIn(partition_mapping=dg.AllPartitionMapping()),
            },
        )
        def summary(upstream):
            # upstream arrives as a dict keyed by MultiPartitionKey with all partitions.
            return dict(upstream)

        defs = dg.Definitions(assets=[upstream, summary])
        instance = dg.DagsterInstance.ephemeral(tempdir=tempfile.mkdtemp())
        job = defs.get_implicit_global_asset_job_def()

        historical_summary_run = None
        historical_upstream_runs = {}
        for key in (
            dg.MultiPartitionKey({"dim": "x", "date": "2024-01-01"}),
            dg.MultiPartitionKey({"dim": "y", "date": "2024-01-01"}),
        ):
            # Select only the upstream asset: the implicit global job would otherwise
            # also execute summary, whose AllPartitionMapping input needs every
            # partition to exist already.
            result = job.execute_in_process(
                instance=instance,
                partition_key=key,
                asset_selection=[dg.AssetKey("upstream")],
            )
            assert result.success
            historical_upstream_runs[str(key)] = result.run_id

        # The summary loads every upstream partition in one input.
        summary_result = job.execute_in_process(
            instance=instance, asset_selection=[dg.AssetKey("summary")]
        )
        assert summary_result.success
        historical_summary_run = summary_result.run_id

        # Rewrite one upstream partition.
        holder[("x", "2024-01-01")] = 99
        job.execute_in_process(
            instance=instance,
            partition_key=dg.MultiPartitionKey({"dim": "x", "date": "2024-01-01"}),
            asset_selection=[dg.AssetKey("upstream")],
        )

        # Replay the summary: every partition of the multi-partition input must be
        # bound to the versions the historical summary run read.
        replay = job.execute_in_process(
            instance=instance,
            tags={REPLAY_TAG: historical_summary_run},
            asset_selection=[dg.AssetKey("summary")],
        )
        assert replay.success
        summary_value = replay.output_for_node("summary")
        assert all(v == 1 for v in summary_value.values())

        manifest = _load_manifest(tmpdir)
        replay_bindings = _input_bindings(manifest, replay.run_id)[_ak("upstream")]
        for key, run_id in historical_upstream_runs.items():
            assert replay_bindings[key] == run_id


def test_op_outputs_remain_unversioned_even_when_versioning_enabled():
    # Versioning applies to assets; op outputs keep their run-scoped paths so jobs
    # built from plain ops are unaffected by the flag.
    with tempfile.TemporaryDirectory() as tmpdir:
        io_manager = dg.FilesystemIOManager(base_dir=tmpdir, versioned=True)

        @dg.op
        def op_a():
            return [1, 2, 3]

        @dg.op
        def op_b(df):
            return df

        @dg.job(resource_defs={"io_manager": io_manager})
        def op_job():
            op_b(op_a())

        result = op_job.execute_in_process(
            instance=dg.DagsterInstance.ephemeral(tempdir=tempfile.mkdtemp())
        )
        assert result.success
        canonical = os.path.join(tmpdir, result.run_id, "op_a", "result")
        assert os.path.isfile(canonical)
        # Plain ops carry no asset key, so versioning never engages: no manifest and
        # no .versions tree are created for this job.
        assert not os.path.exists(os.path.join(tmpdir, MANIFEST_FILENAME))
        assert not os.path.exists(os.path.join(tmpdir, VERSIONS_DIRNAME))


def test_replay_run_id_config_is_overridden_by_run_tag():
    with tempfile.TemporaryDirectory() as tmpdir:
        # A static replay id is configured, but the run tag points at the run that
        # actually exists. The tag must win; otherwise this run would fail against
        # the bogus static id.
        io_manager = dg.FilesystemIOManager(
            base_dir=tmpdir, versioned=True, replay_run_id="stale-static-id"
        )

        @dg.asset(io_manager_def=io_manager)
        def upstream():
            return 1

        @dg.asset(io_manager_def=io_manager)
        def report(upstream):
            return upstream

        defs = dg.Definitions(assets=[upstream, report])
        instance = dg.DagsterInstance.ephemeral(tempdir=tempfile.mkdtemp())
        job = defs.get_implicit_global_asset_job_def()

        historical = job.execute_in_process(instance=instance)
        replay = job.execute_in_process(
            instance=instance,
            tags={REPLAY_TAG: historical.run_id},
            asset_selection=[dg.AssetKey("report")],
        )
        assert replay.success
        manifest = _load_manifest(tmpdir)
        assert _input_bindings(manifest, replay.run_id)[_ak("upstream")][""] == historical.run_id


# Module-level worker so it can be pickled for the spawn multiprocessing context.
def _manifest_worker(args: tuple[str, int, int]) -> None:
    from dagster._core.storage.versioned_io_manager import VersionManifest
    from upath import UPath

    base_dir, worker_id, writes_per_worker = args
    manifest = VersionManifest(UPath(base_dir) / MANIFEST_FILENAME)
    for i in range(writes_per_worker):
        manifest.record_output(
            run_id=f"run-{worker_id}-{i}",
            asset_key='["asset"]',
            partition=None,
            version_run_id=f"run-{worker_id}-{i}",
            path=f"/path/{worker_id}/{i}",
            logical_path="/logical/asset",
        )


def test_manifest_survives_cross_process_concurrent_writes():
    # The multiprocess executor runs steps in separate worker processes that each
    # open their own IO manager (and thus their own VersionManifest). Every
    # read-modify-write must be serialized by the fcntl lock so no update is lost.
    import multiprocessing as mp

    worker_count = 4
    writes_per_worker = 25
    with tempfile.TemporaryDirectory() as tmpdir:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=worker_count) as pool:
            pool.map(
                _manifest_worker,
                [(tmpdir, worker_id, writes_per_worker) for worker_id in range(worker_count)],
            )

        manifest = _load_manifest(tmpdir)
        recorded_runs = manifest["runs"]
        expected = worker_count * writes_per_worker
        assert len(recorded_runs) == expected
        # Every single version path must be present; a lost update would show up as
        # either a missing run entry or a clobbered one.
        for worker_id in range(worker_count):
            for i in range(writes_per_worker):
                run_id = f"run-{worker_id}-{i}"
                assert run_id in recorded_runs
                slot = recorded_runs[run_id]["outputs"]
                assert slot['["asset"]::']["path"] == f"/path/{worker_id}/{i}"


# Module-level assets so the definition is reconstructable across worker processes.
@dg.asset
def mp_asset_a():
    return "a"


@dg.asset
def mp_asset_b():
    return "b"


@dg.asset
def mp_asset_c(mp_asset_a, mp_asset_b):
    return mp_asset_a + mp_asset_b


def _build_multiprocess_job():
    # base_dir is supplied at launch time through run_config so the function stays
    # argument-free and therefore reconstructable in worker processes.
    io_manager = dg.FilesystemIOManager(versioned=True)
    return dg.define_asset_job(
        "mp_versioned_job",
        [mp_asset_a, mp_asset_b, mp_asset_c],
    ).resolve(
        asset_graph=AssetGraph.from_assets([mp_asset_a, mp_asset_b, mp_asset_c]),
        resource_defs={"io_manager": io_manager},
        default_executor_def=dg.multiprocess_executor,
    )


def test_versioned_io_manager_with_multiprocess_executor(tmp_path):
    # End-to-end coverage: separate worker processes each instantiate their own
    # versioned IO manager and update the shared manifest concurrently.
    storage_dir = str(tmp_path / "storage")

    # A persistent (non-ephemeral) instance is required to coordinate worker
    # processes under the multiprocess executor. Both config fields must be passed:
    # run_config replaces (rather than merges into) the resource's field defaults.
    with dg.instance_for_test() as instance:
        result = dg.execute_job(
            dg.reconstructable(_build_multiprocess_job),
            instance=instance,
            run_config={
                "resources": {
                    "io_manager": {"config": {"base_dir": storage_dir, "versioned": True}}
                }
            },
        )
        assert result.success

    # All three assets materialized, each under its own immutable version path, and
    # the manifest captured every output without losing a cross-process write.
    manifest = _load_manifest(storage_dir)
    outputs = manifest["runs"][result.run_id]["outputs"]
    assert len(outputs) == 3
    for asset_name in ("mp_asset_a", "mp_asset_b", "mp_asset_c"):
        slot = outputs[f'["{asset_name}"]::']
        assert os.path.isfile(slot["path"])
