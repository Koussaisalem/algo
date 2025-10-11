from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qcmd_hybrid_framework.qcmd_ecs.core.dynamics import run_reverse_diffusion
from qcmd_hybrid_framework.qcmd_ecs.core.manifold import project_to_tangent_space, retract_to_manifold
from qcmd_hybrid_framework.qcmd_ecs.core.types import DTYPE

AtomicSample = Dict[str, torch.Tensor]
Schedule = Callable[[int], float]
ScoreFn = Callable[[torch.Tensor, int], torch.Tensor]
EnergyGradFn = Callable[[torch.Tensor], torch.Tensor]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark QCMD-ECS diffusion against a Euclidean baseline using oracle scores.",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("data/qm9_micro_5k_enriched.pt"),
        help="Path to the enriched dataset with manifold frames.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/benchmark_cmd_vs_euclidean"),
        help="Directory where per-sample metrics and summary will be stored.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=32,
        help="Number of molecules to evaluate (random subset).",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=30,
        help="Number of reverse-diffusion steps for each generator.",
    )
    parser.add_argument(
        "--noise-scale",
        type=float,
        default=0.15,
        help="Standard deviation of the Gaussian noise added to the ground-truth frame for the initial state.",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=0.05,
        help="Constant step size for both generators.",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=0.02,
        help="Constant noise scale per diffusion step.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.0,
        help="Constant gamma schedule weight for the energy-gradient term.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Base random seed for subset selection and noise sampling.",
    )
    return parser.parse_args()


def load_dataset(path: Path) -> Sequence[AtomicSample]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    data = torch.load(path, map_location="cpu")
    if not isinstance(data, Sequence):
        raise TypeError("Expected dataset to be a sequence of dict samples")
    return data


def choose_indices(num_items: int, num_samples: int, seed: int) -> List[int]:
    if num_samples > num_items:
        raise ValueError("num_samples cannot exceed available dataset size")
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(num_items, generator=generator).tolist()
    return permutation[:num_samples]


def constant_schedule(value: float) -> Schedule:
    return lambda _t: float(value)


def oracle_score_fn(target: torch.Tensor, weight: float = 1.0) -> ScoreFn:
    target = target.clone()

    def score(U_t: torch.Tensor, _t: int) -> torch.Tensor:
        return weight * (target - U_t)

    return score


def zero_energy_gradient(_U_t: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(_U_t)


def orthogonality_error(U: torch.Tensor) -> float:
    k = U.shape[1]
    identity = torch.eye(k, dtype=U.dtype, device=U.device)
    gram = U.transpose(-1, -2) @ U
    return torch.linalg.norm(gram - identity).item()


def frobenius_error(U: torch.Tensor, target: torch.Tensor) -> float:
    return torch.linalg.norm(U - target).item()


def cosine_alignment(U: torch.Tensor, target: torch.Tensor) -> float:
    numerator = torch.sum(U * target)
    denom = math.sqrt(torch.sum(U * U).item() * torch.sum(target * target).item())
    if denom == 0.0:
        return float("nan")
    return float(numerator.item() / denom)


def run_euclidean_diffusion(
    U_T: torch.Tensor,
    score_model: ScoreFn,
    energy_gradient_model: EnergyGradFn,
    gamma_schedule: Schedule,
    eta_schedule: Schedule,
    tau_schedule: Schedule,
    num_steps: int,
    seed: int,
) -> torch.Tensor:
    torch.manual_seed(seed)
    U_t = U_T.clone()
    for t in range(num_steps, 0, -1):
        s_t = score_model(U_t, t)
        grad_E_t = energy_gradient_model(U_t)
        noise = torch.randn_like(U_t, dtype=DTYPE)
        update = s_t + gamma_schedule(t) * grad_E_t
        U_t = U_t - eta_schedule(t) * update + tau_schedule(t) * noise
    return U_t


@dataclass
class SampleMetrics:
    index: int
    rank: int
    manifold_fro_error: float
    euclidean_fro_error: float
    manifold_orth_error: float
    euclidean_orth_error: float
    manifold_alignment: float
    euclidean_alignment: float
    manifold_time_ms: float
    euclidean_time_ms: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "index": self.index,
            "rank": self.rank,
            "manifold_fro_error": self.manifold_fro_error,
            "euclidean_fro_error": self.euclidean_fro_error,
            "manifold_orth_error": self.manifold_orth_error,
            "euclidean_orth_error": self.euclidean_orth_error,
            "manifold_alignment": self.manifold_alignment,
            "euclidean_alignment": self.euclidean_alignment,
            "manifold_time_ms": self.manifold_time_ms,
            "euclidean_time_ms": self.euclidean_time_ms,
        }


def benchmark_sample(
    sample: AtomicSample,
    args: argparse.Namespace,
    base_seed: int,
    eta_schedule: Schedule,
    tau_schedule: Schedule,
    gamma_schedule: Schedule,
) -> SampleMetrics:
    frame = sample["manifold_frame"]["frame"].to(dtype=DTYPE)
    if frame.ndim != 2:
        raise ValueError("Frame tensor must be 2-dimensional")
    target = frame.contiguous()
    rank_tensor = sample["manifold_frame"]["rank"]
    rank = int(rank_tensor.item()) if isinstance(rank_tensor, torch.Tensor) else int(rank_tensor)

    generator = torch.Generator().manual_seed(base_seed)
    noise = torch.randn(target.shape, generator=generator, dtype=DTYPE)
    noisy_ambient = target + args.noise_scale * noise
    U_T_manifold = retract_to_manifold(noisy_ambient)
    U_T_euclidean = noisy_ambient.clone()

    score_model = oracle_score_fn(target)
    energy_model = zero_energy_gradient

    start = time.perf_counter()
    manifold_result = run_reverse_diffusion(
        U_T=U_T_manifold,
        score_model=score_model,
        energy_gradient_model=energy_model,
        gamma_schedule=gamma_schedule,
        eta_schedule=eta_schedule,
        tau_schedule=tau_schedule,
        num_steps=args.num_steps,
        seed=base_seed,
    )
    manifold_time_ms = (time.perf_counter() - start) * 1000.0

    start = time.perf_counter()
    euclidean_result = run_euclidean_diffusion(
        U_T=U_T_euclidean,
        score_model=score_model,
        energy_gradient_model=energy_model,
        gamma_schedule=gamma_schedule,
        eta_schedule=eta_schedule,
        tau_schedule=tau_schedule,
        num_steps=args.num_steps,
        seed=base_seed,
    )
    euclidean_time_ms = (time.perf_counter() - start) * 1000.0

    return SampleMetrics(
        index=int(sample["index"]) if "index" in sample else -1,
        rank=rank,
        manifold_fro_error=frobenius_error(manifold_result, target),
        euclidean_fro_error=frobenius_error(euclidean_result, target),
        manifold_orth_error=orthogonality_error(manifold_result),
        euclidean_orth_error=orthogonality_error(euclidean_result),
        manifold_alignment=cosine_alignment(manifold_result, target),
        euclidean_alignment=cosine_alignment(euclidean_result, target),
        manifold_time_ms=manifold_time_ms,
        euclidean_time_ms=euclidean_time_ms,
    )


def summarise(metrics: Iterable[SampleMetrics]) -> Dict[str, float]:
    metrics_list = list(metrics)
    if not metrics_list:
        return {}
    def mean(attr: str) -> float:
        values = [getattr(item, attr) for item in metrics_list]
        return float(sum(values) / len(values))

    summary = {
        "count": len(metrics_list),
        "manifold_fro_error_mean": mean("manifold_fro_error"),
        "euclidean_fro_error_mean": mean("euclidean_fro_error"),
        "manifold_orth_error_mean": mean("manifold_orth_error"),
        "euclidean_orth_error_mean": mean("euclidean_orth_error"),
        "manifold_alignment_mean": mean("manifold_alignment"),
        "euclidean_alignment_mean": mean("euclidean_alignment"),
        "manifold_time_ms_mean": mean("manifold_time_ms"),
        "euclidean_time_ms_mean": mean("euclidean_time_ms"),
    }
    return summary


def write_outputs(
    metrics: List[SampleMetrics],
    summary: Dict[str, float],
    args: argparse.Namespace,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "per_sample_metrics.json"
    summary_path = output_dir / "summary.json"
    report_path = output_dir / "report.md"

    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump([item.as_dict() for item in metrics], handle, indent=2)

    with summary_path.open("w", encoding="utf-8") as handle:
        payload = {
            "summary": summary,
            "config": {
                "num_samples": args.num_samples,
                "num_steps": args.num_steps,
                "noise_scale": args.noise_scale,
                "eta": args.eta,
                "tau": args.tau,
                "gamma": args.gamma,
                "seed": args.seed,
                "dataset_path": str(args.dataset_path),
            },
        }
        json.dump(payload, handle, indent=2)

    with report_path.open("w", encoding="utf-8") as handle:
        handle.write("# CMD-ECS vs Euclidean Benchmark\n\n")
        handle.write("## Configuration\n\n")
        handle.write(
            "- Samples: {count}\n- Steps: {steps}\n- Noise scale: {noise:.4f}\n- Eta: {eta:.4f}\n- Tau: {tau:.4f}\n- Gamma: {gamma:.4f}\n- Seed: {seed}\n\n".format(
                count=summary.get("count", 0),
                steps=args.num_steps,
                noise=args.noise_scale,
                eta=args.eta,
                tau=args.tau,
                gamma=args.gamma,
                seed=args.seed,
            )
        )

        handle.write("## Mean Metrics\n\n")
        handle.write("| Metric | Manifold | Euclidean |\n")
        handle.write("| --- | ---: | ---: |\n")
        handle.write(
            "| Frobenius error | {mf:.6f} | {ef:.6f} |\n".format(
                mf=summary.get("manifold_fro_error_mean", float("nan")),
                ef=summary.get("euclidean_fro_error_mean", float("nan")),
            )
        )
        handle.write(
            "| Orthogonality error | {mo:.6f} | {eo:.6f} |\n".format(
                mo=summary.get("manifold_orth_error_mean", float("nan")),
                eo=summary.get("euclidean_orth_error_mean", float("nan")),
            )
        )
        handle.write(
            "| Cosine alignment | {ma:.6f} | {ea:.6f} |\n".format(
                ma=summary.get("manifold_alignment_mean", float("nan")),
                ea=summary.get("euclidean_alignment_mean", float("nan")),
            )
        )
        handle.write(
            "| Runtime (ms) | {mt:.3f} | {et:.3f} |\n".format(
                mt=summary.get("manifold_time_ms_mean", float("nan")),
                et=summary.get("euclidean_time_ms_mean", float("nan")),
            )
        )

        handle.write("\n## Notes\n\n")
        handle.write(
            "- Oracle score pulls states toward the dataset frame; no trained score model is required.\n"
        )
        handle.write(
            "- Euclidean baseline omits tangent projection and retraction, highlighting orthogonality drift.\n"
        )


def main() -> None:
    args = parse_args()
    torch.set_default_dtype(DTYPE)

    samples = load_dataset(args.dataset_path)
    indices = choose_indices(len(samples), args.num_samples, args.seed)

    eta_schedule = constant_schedule(args.eta)
    tau_schedule = constant_schedule(args.tau)
    gamma_schedule = constant_schedule(args.gamma)

    metrics: List[SampleMetrics] = []
    for offset, idx in enumerate(indices):
        sample = samples[idx]
        base_seed = args.seed + offset
        metrics.append(
            benchmark_sample(
                sample=sample,
                args=args,
                base_seed=base_seed,
                eta_schedule=eta_schedule,
                tau_schedule=tau_schedule,
                gamma_schedule=gamma_schedule,
            )
        )

    summary = summarise(metrics)
    write_outputs(metrics, summary, args, args.output_dir)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
