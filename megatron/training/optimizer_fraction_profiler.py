# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Low-overhead CUDA-event timing for optimizer/step time fractions."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed


@dataclass
class _IterationEvents:
    iteration: int
    step_start: torch.cuda.Event
    optimizer_start: torch.cuda.Event | None = None
    optimizer_end: torch.cuda.Event | None = None
    step_end: torch.cuda.Event | None = None


@dataclass
class _QRCallEvents:
    iteration: int
    requested_backend: str
    actual_backend: str
    rows: int
    columns: int
    dtype: str
    start: torch.cuda.Event
    end: torch.cuda.Event | None = None


class OptimizerFractionProfiler:
    """Record rank-local train-step and optimizer durations with CUDA events.

    Events are recorded only inside the configured inclusive iteration window.
    CUDA is synchronized once, after the final event in the window, before all
    rank-local measurements are written atomically to CSV.
    """

    def __init__(
        self,
        start_iteration: int,
        end_iteration: int,
        output_dir: str,
        global_batch_size: int,
        profile_soap_qr: bool = False,
    ):
        if start_iteration < 1:
            raise ValueError("optimizer profiling start iteration must be at least 1")
        if end_iteration < start_iteration:
            raise ValueError("optimizer profiling end iteration must not precede its start")
        if not output_dir:
            raise ValueError("--profile-optimizer-output-dir is required when profiling is enabled")

        self.start_iteration = start_iteration
        self.end_iteration = end_iteration
        self.output_dir = Path(output_dir)
        self.global_batch_size = global_batch_size
        self.rank = torch.distributed.get_rank()
        self.world_size = torch.distributed.get_world_size()
        self.device = torch.cuda.current_device()
        self.profile_soap_qr = profile_soap_qr
        self._events: list[_IterationEvents] = []
        self._qr_events: list[_QRCallEvents] = []
        self._active: _IterationEvents | None = None
        self._written = False
        self._eig_utils = None
        if self.profile_soap_qr:
            from emerging_optimizers.utils import eig as eig_utils

            eig_utils.set_qr_timing_observer(self)
            self._eig_utils = eig_utils

    @classmethod
    def from_args(cls, args):
        profile_optimizer_fraction = getattr(args, "profile_optimizer_fraction", False)
        profile_soap_qr = getattr(args, "profile_soap_qr", False)
        if profile_soap_qr and not profile_optimizer_fraction:
            raise ValueError("--profile-soap-qr requires --profile-optimizer-fraction")
        if not profile_optimizer_fraction:
            return None
        if profile_soap_qr and args.optimizer != "soap":
            raise ValueError("--profile-soap-qr requires --optimizer soap")
        if args.profile_optimizer_end_iteration > args.train_iters:
            raise ValueError(
                "optimizer profiling end iteration "
                f"{args.profile_optimizer_end_iteration} exceeds train-iters {args.train_iters}"
            )
        return cls(
            start_iteration=args.profile_optimizer_start_iteration,
            end_iteration=args.profile_optimizer_end_iteration,
            output_dir=args.profile_optimizer_output_dir,
            global_batch_size=args.global_batch_size,
            profile_soap_qr=profile_soap_qr,
        )

    def _in_window(self, iteration: int) -> bool:
        return self.start_iteration <= iteration <= self.end_iteration

    @staticmethod
    def _new_event() -> torch.cuda.Event:
        return torch.cuda.Event(enable_timing=True)

    def start_step(self, iteration: int) -> None:
        if not self._in_window(iteration):
            return
        if self._active is not None:
            raise RuntimeError("optimizer profiler already has an active training step")
        event = self._new_event()
        event.record()
        self._active = _IterationEvents(iteration=iteration, step_start=event)

    def start_optimizer(self, iteration: int) -> None:
        if not self._in_window(iteration):
            return
        if self._active is None or self._active.iteration != iteration:
            raise RuntimeError("optimizer profiler did not observe the matching step start")
        event = self._new_event()
        event.record()
        self._active.optimizer_start = event

    def stop_optimizer(self, iteration: int) -> None:
        if not self._in_window(iteration):
            return
        if self._active is None or self._active.iteration != iteration:
            raise RuntimeError("optimizer profiler did not observe the matching step start")
        if self._active.optimizer_start is None:
            raise RuntimeError("optimizer profiler did not observe optimizer start")
        event = self._new_event()
        event.record()
        self._active.optimizer_end = event

    def start_qr(
        self,
        *,
        requested_backend: str,
        actual_backend: str,
        matrix: torch.Tensor,
    ) -> _QRCallEvents | None:
        """Record the start of one SOAP QR call on its current CUDA stream."""
        if not self.profile_soap_qr:
            return None
        if self._active is None or self._active.optimizer_start is None:
            return None
        event = self._new_event()
        event.record()
        return _QRCallEvents(
            iteration=self._active.iteration,
            requested_backend=requested_backend,
            actual_backend=actual_backend,
            rows=matrix.shape[0],
            columns=matrix.shape[1],
            dtype=str(matrix.dtype).removeprefix("torch."),
            start=event,
        )

    def stop_qr(self, token: object | None) -> None:
        """Record the end of a SOAP QR call started by :meth:`start_qr`."""
        if token is None:
            return
        if not isinstance(token, _QRCallEvents):
            raise TypeError(f"unexpected SOAP QR timing token: {type(token)!r}")
        event = self._new_event()
        event.record()
        token.end = event
        self._qr_events.append(token)

    def stop_step(self, iteration: int) -> None:
        if not self._in_window(iteration):
            return
        if self._active is None or self._active.iteration != iteration:
            raise RuntimeError("optimizer profiler did not observe the matching step start")
        if self._active.optimizer_end is None:
            raise RuntimeError("optimizer profiler did not observe optimizer end")
        event = self._new_event()
        event.record()
        self._active.step_end = event
        self._events.append(self._active)
        self._active = None
        if iteration == self.end_iteration:
            self._write_results()

    def _write_results(self) -> None:
        if self._written:
            return
        torch.cuda.synchronize()
        expected_count = self.end_iteration - self.start_iteration + 1
        if len(self._events) != expected_count:
            raise RuntimeError(
                f"optimizer profiler collected {len(self._events)} iterations; expected {expected_count}"
            )

        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / f"timing_rank{self.rank}.csv"
        temporary_path = output_path.with_suffix(".csv.tmp")
        fieldnames = [
            "iteration",
            "rank",
            "world_size",
            "cuda_device",
            "global_batch_size",
            "step_ms",
            "optimizer_ms",
            "optimizer_fraction",
        ]
        with temporary_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for events in self._events:
                assert events.optimizer_start is not None
                assert events.optimizer_end is not None
                assert events.step_end is not None
                step_ms = events.step_start.elapsed_time(events.step_end)
                optimizer_ms = events.optimizer_start.elapsed_time(events.optimizer_end)
                writer.writerow(
                    {
                        "iteration": events.iteration,
                        "rank": self.rank,
                        "world_size": self.world_size,
                        "cuda_device": self.device,
                        "global_batch_size": self.global_batch_size,
                        "step_ms": f"{step_ms:.6f}",
                        "optimizer_ms": f"{optimizer_ms:.6f}",
                        "optimizer_fraction": f"{optimizer_ms / step_ms:.9f}",
                    }
                )
        temporary_path.replace(output_path)
        if self.profile_soap_qr:
            self._write_qr_results()
            assert self._eig_utils is not None
            self._eig_utils.set_qr_timing_observer(None)
        self._written = True

    def _write_qr_results(self) -> None:
        expected_iterations = set(range(self.start_iteration, self.end_iteration + 1))
        observed_iterations = {events.iteration for events in self._qr_events}
        if observed_iterations != expected_iterations:
            raise RuntimeError(
                "SOAP QR profiler observed iterations "
                f"{sorted(observed_iterations)}; expected {sorted(expected_iterations)}"
            )
        if any(events.end is None for events in self._qr_events):
            raise RuntimeError("SOAP QR profiler has an unfinished QR call")

        output_path = self.output_dir / f"qr_timing_rank{self.rank}.csv"
        temporary_path = output_path.with_suffix(".csv.tmp")
        fieldnames = [
            "iteration",
            "rank",
            "world_size",
            "cuda_device",
            "global_batch_size",
            "call_index",
            "requested_backend",
            "actual_backend",
            "rows",
            "columns",
            "dtype",
            "qr_ms",
        ]
        call_counts: dict[int, int] = {}
        with temporary_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for events in self._qr_events:
                call_index = call_counts.get(events.iteration, 0)
                call_counts[events.iteration] = call_index + 1
                assert events.end is not None
                writer.writerow(
                    {
                        "iteration": events.iteration,
                        "rank": self.rank,
                        "world_size": self.world_size,
                        "cuda_device": self.device,
                        "global_batch_size": self.global_batch_size,
                        "call_index": call_index,
                        "requested_backend": events.requested_backend,
                        "actual_backend": events.actual_backend,
                        "rows": events.rows,
                        "columns": events.columns,
                        "dtype": events.dtype,
                        "qr_ms": f"{events.start.elapsed_time(events.end):.6f}",
                    }
                )
        temporary_path.replace(output_path)
