"""Version retention and historical replay support for file-backed IO managers.

The default :class:`~dagster._core.storage.upath_io_manager.UPathIOManager` layout
stores each (non-partitioned) asset at a single path, so a re-materialization
overwrites the previous one. Once an upstream partition has been rewritten, an old
downstream report can no longer be reproduced: re-running the report reads the new
upstream data instead of the data the original run actually consumed.

This module adds an *opt-in* versioned layer on top of the same path machinery:

* when versioning is enabled, every materialization is written to a new, immutable
  version path (``.../<asset>/.versions/<run_id>``) instead of overwriting the
  canonical path;
* a manifest next to the storage root records every materialization and every input
  version actually read by each run, including the individual partitions of a
  multi-partition load;
* a replay run is bound to a historical run id. Inputs are served from the versions
  the bound run recorded, except that partitions re-materialized by the replay run
  itself are served from the replay run's own new versions;
* if a bound version is missing on disk (never materialized or since deleted), the
  load terminates with :class:`DagsterVersionReplayError` instead of falling back to
  the latest data;
* outputs of a replay are always stored as new versions, never overwriting the old
  ones.

When versioning is disabled (the default), none of this code runs and the IO
manager behaves exactly as before.
"""

import json
import os
import tempfile
from contextlib import contextmanager
from threading import Lock
from typing import TYPE_CHECKING, Any

from dagster._core.errors import DagsterInvariantViolationError, DagsterVersionReplayError

if TYPE_CHECKING:
    from upath import UPath

try:
    # POSIX only; on Windows we fall back to the process-local threading lock.
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX platforms
    fcntl = None

# Run tag used to bind a run to the historical run whose input versions it must
# replay. Lives in the public ``dagster/`` tag namespace so it can be set through the
# normal run launch configuration, alongside tags such as the backfill id.
VERSION_REPLAY_RUN_ID_TAG = "dagster/version_replay_run_id"

MANIFEST_FILENAME = ".version_manifest.json"
VERSIONS_DIRNAME = ".versions"

# Bump when the on-disk manifest format changes in an incompatible way.
MANIFEST_VERSION = 1


class VersionManifest:
    """JSON-backed record of every materialization and input binding per run.

    Shape (all keys are strings)::

        {
          "manifest_version": 1,
          "runs": {
            "<run_id>": {
              "outputs": {
                "<asset_key>": {
                  "partition": "<partition key or null>",
                  "version_run_id": "<run that wrote the bytes>",
                  "path": "<version path actually written>",
                  "logical_path": "<canonical (non-versioned) path>"
                },
                ...
              },
              "inputs": {
                "<asset_key>": {
                  "<partition key or empty string>": "<version run id actually read>"
                }
              }
            }
          }
        }

    The manifest is the single source of truth for "which version did run X read".
    Every read-modify-write holds an exclusive lock: a process-local threading lock
    for in-process steps and, on a local filesystem, a ``fcntl`` lock on a sibling
    lock file so separate worker processes (multiprocess executor) cannot clobber
    each other's updates. The final write is an atomic temp-file replace.

    On non-local (object-store) backends there is no POSIX file lock, so only the
    in-process lock applies and the replace is the backend's own write. Versioned
    replay is therefore strongly consistent on local filesystems; object stores
    should rely on their atomic put-if-absent semantics when available.
    """

    def __init__(self, path: "UPath"):
        self._path = path
        # Serializes concurrent steps inside one process.
        self._lock = Lock()
        # Lazily opened, kept open for the manifest's lifetime so the fcntl lock is
        # tied to a stable file descriptor.
        self._lock_file = None

    @contextmanager
    def _locked(self):
        """Acquire both the in-process lock and, when available, a cross-process
        exclusive file lock for the duration of a read-modify-write cycle.
        """
        self._lock.acquire()
        try:
            if fcntl is not None and _is_local(self._path.parent):
                self._path.parent.mkdir(parents=True, exist_ok=True)
                lock_path = os.fspath(self._path) + ".lock"
                if self._lock_file is None:
                    self._lock_file = open(lock_path, "a+", encoding="utf-8")
                fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_EX)
            yield
        finally:
            if self._lock_file is not None and fcntl is not None:
                fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_UN)
            self._lock.release()

    # ------------------------------------------------------------------ I/O
    def _read(self) -> dict[str, Any]:
        try:
            with self._path.open("r") as f:
                data = json.load(f)
        except FileNotFoundError:
            return {"manifest_version": MANIFEST_VERSION, "runs": {}}
        if not isinstance(data, dict) or "runs" not in data:
            raise DagsterVersionReplayError(
                f"Version manifest at {self._path} is corrupted: expected an object with a"
                " 'runs' mapping. Refusing to guess input versions; inspect or remove the"
                " manifest before retrying."
            )
        return data

    def _write_atomic(self, data: dict[str, Any]) -> None:
        parent = self._path.parent
        parent.mkdir(parents=True, exist_ok=True)
        # Write to a temp file in the same directory, then atomically replace. On the
        # local filesystem os.replace is atomic; fsspec backends implement the same
        # primitive for their object stores.
        if _is_local(parent):
            fd, tmp_name = tempfile.mkstemp(dir=os.fspath(parent), prefix=".manifest-")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as tmp:
                    json.dump(data, tmp, indent=2, sort_keys=True)
                os.replace(tmp_name, os.fspath(self._path))
            except BaseException:
                os.unlink(tmp_name)
                raise
        else:
            # Remote backends have no local temp dir; stream JSON then let the backend
            # replace handle the swap.
            with self._path.open("w") as f:
                json.dump(data, f, indent=2, sort_keys=True)

    # ------------------------------------------------------------- mutation
    def record_output(
        self,
        run_id: str,
        asset_key: str,
        partition: str | None,
        version_run_id: str,
        path: str,
        logical_path: str,
    ) -> None:
        with self._locked():
            data = self._read()
            run_entry = data["runs"].setdefault(run_id, {"outputs": {}, "inputs": {}})
            run_entry.setdefault("outputs", {})
            # One (run, asset, partition) produces exactly one version. A retry of the
            # same step re-records the same run id; the version path is deterministic,
            # so the entry is idempotent.
            run_entry["outputs"][_output_slot(asset_key, partition)] = {
                "asset_key": asset_key,
                "partition": partition,
                "version_run_id": version_run_id,
                "path": path,
                "logical_path": logical_path,
            }
            self._write_atomic(data)

    def record_input(
        self,
        run_id: str,
        asset_key: str,
        partition: str | None,
        version_run_id: str,
    ) -> None:
        with self._locked():
            data = self._read()
            run_entry = data["runs"].setdefault(run_id, {"outputs": {}, "inputs": {}})
            inputs = run_entry.setdefault("inputs", {})
            asset_inputs = inputs.setdefault(asset_key, {})
            asset_inputs[partition if partition is not None else ""] = version_run_id
            self._write_atomic(data)

    # -------------------------------------------------------------- lookup
    def get_run(self, run_id: str) -> dict[str, Any] | None:
        return self._read()["runs"].get(run_id)

    def iter_output_slots(self) -> list[tuple[str, str, str | None, str]]:
        """Return every recorded output slot across all runs as
        ``(run_id, asset_key, partition, version_run_id)`` in recording order. Used to
        resolve the latest retained version for an unbound read.
        """
        return [
            (run_id, output["asset_key"], output["partition"], output["version_run_id"])
            for run_id, run_entry in self._read()["runs"].items()
            for output in run_entry.get("outputs", {}).values()
        ]

    def get_output_version(self, run_id: str, asset_key: str, partition: str | None) -> str | None:
        """Return the version run id that ``run_id`` produced for the given slot, or
        None if that run never materialized it.
        """
        run_entry = self.get_run(run_id)
        if run_entry is None:
            return None
        entry = run_entry.get("outputs", {}).get(_output_slot(asset_key, partition))
        return entry["version_run_id"] if entry else None

    def get_input_binding(self, run_id: str, asset_key: str, partition: str | None) -> str | None:
        """Return the version run id that ``run_id`` actually read for this input
        slot, or None if the run recorded no such input.
        """
        run_entry = self.get_run(run_id)
        if run_entry is None:
            return None
        return (
            run_entry.get("inputs", {})
            .get(asset_key, {})
            .get(partition if partition is not None else "")
        )


def _is_local(path: "UPath") -> bool:
    protocol = getattr(path, "protocol", "")
    return protocol in ("", "file")


def _output_slot(asset_key: str, partition: str | None) -> str:
    return f"{asset_key}::{partition if partition is not None else ''}"


def resolve_replay_run_id(context: Any, configured: str | None) -> str | None:
    """Determine the historical run id to replay, if any.

    A run tag (``VERSION_REPLAY_RUN_ID_TAG``) takes precedence over static resource
    configuration so a single IO manager definition can serve both normal and
    replay runs. Falls back to None when no step context is available (unit tests
    building contexts directly).
    """
    try:
        tags = context.step_context.dagster_run.tags
    except DagsterInvariantViolationError:
        # build_input_context/build_output_context without a step_context (unit tests)
        # raise when the step context is accessed.
        tags = None
    if tags and VERSION_REPLAY_RUN_ID_TAG in tags:
        return tags[VERSION_REPLAY_RUN_ID_TAG]
    return configured


def raise_missing_version(
    *,
    asset_key: str,
    partition: str | None,
    version_run_id: str | None,
    bound_run_id: str | None,
    path: "UPath",
) -> None:
    target = (
        f"partition '{partition}' of asset '{asset_key}'" if partition else f"asset '{asset_key}'"
    )
    if version_run_id is None:
        raise DagsterVersionReplayError(
            f"Cannot replay inputs for {target}: bound historical run"
            f" {bound_run_id!r} recorded no input version for this slot, and the current"
            " run has not re-materialized it. Re-materialize the missing upstream output in"
            " this run, or choose a historical run that consumed a retained version. The"
            " load was terminated to avoid silently reading latest data."
        )
    raise DagsterVersionReplayError(
        f"Cannot replay inputs for {target}: the required version produced by run"
        f" {version_run_id!r} is missing at {path}. The replay was terminated rather than"
        " falling back to the latest materialization. Re-materialize the upstream output in"
        " this run, or restore the missing version files."
    )
