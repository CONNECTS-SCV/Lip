"""CLI entry point - python -m lip"""

from __future__ import annotations

import argparse
import logging
import sys

from lip import __version__
from lip.config import LipConfig


def main():
    parser = argparse.ArgumentParser(
        prog="lip",
        description="Lip - RL-based molecular generation and optimization",
    )
    parser.add_argument("--version", action="version", version=f"lip {__version__}")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- run ---
    run_parser = subparsers.add_parser("run", help="Run full optimization pipeline")
    _add_common_args(run_parser)
    _add_generator_args(run_parser)
    _add_optimization_args(run_parser)
    _add_docking_args(run_parser)
    _add_constraint_args(run_parser)
    _add_path_args(run_parser)
    _add_synthesis_args(run_parser)
    run_parser.add_argument("--resume", type=str, default=None, help="Resume from run directory")

    # --- score ---
    score_parser = subparsers.add_parser("score", help="Score SMILES against constraints")
    _add_common_args(score_parser)
    _add_docking_args(score_parser)
    _add_constraint_args(score_parser)
    _add_path_args(score_parser)
    score_parser.add_argument("--smiles", type=str, nargs="+", required=True, help="SMILES to score")

    # --- synthesis ---
    synth_parser = subparsers.add_parser("synthesis", help="Analyze synthesis routes for molecules")
    _add_common_args(synth_parser)
    _add_path_args(synth_parser)
    _add_synthesis_args(synth_parser)
    synth_parser.add_argument("--smiles", type=str, nargs="+", required=False,
                              help="SMILES to analyze")
    synth_parser.add_argument("--input-csv", dest="input_csv", type=str, default=None,
                              help="CSV file with SMILES column")
    synth_parser.add_argument("--smiles-column", dest="smiles_column", type=str, default="smiles",
                              help="Column name for SMILES in CSV (default: smiles)")

    # --- pocket2mol ---
    p2m_parser = subparsers.add_parser("pocket2mol", help="Generate reference molecules")
    _add_common_args(p2m_parser)
    _add_path_args(p2m_parser)
    p2m_parser.add_argument("--n-samples", type=int, default=3)
    p2m_parser.add_argument("--bbox-size", type=float, default=23.0)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    # Setup logging
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Build config: defaults → YAML → CLI
    config = LipConfig()
    if args.config:
        config = LipConfig.from_yaml(args.config)
    config.merge_cli(_args_to_dict(args))

    if args.command == "run":
        _cmd_run(config, args)
    elif args.command == "score":
        _cmd_score(config, args)
    elif args.command == "synthesis":
        _cmd_synthesis(config, args)
    elif args.command == "pocket2mol":
        _cmd_pocket2mol(config, args)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def _cmd_run(config: LipConfig, args):
    from lip.runner import run

    result = run(config, resume_dir=args.resume)
    print(f"\nOptimization complete.")
    print(f"  Output: {result.output_dir}")
    print(f"  Best score: {result.best_score:.4f}")
    print(f"  Total molecules: {result.total_molecules}")
    if result.resumed:
        print(f"  (Resumed from previous run)")


def _cmd_score(config: LipConfig, args):
    from lip.constraints.registry import create_constraint

    smiles_list = args.smiles
    print(f"Scoring {len(smiles_list)} molecule(s)...\n")

    for cc in config.constraints:
        constraint = create_constraint(cc.type, cc.weight, cc.params)
        result = constraint.score(smiles_list)
        print(f"[{cc.type}] (weight={cc.weight})")
        for i, smi in enumerate(smiles_list):
            score = result.scores[i] if i < len(result.scores) else 0.0
            raw = result.raw_values[i] if i < len(result.raw_values) else "-"
            status = "PASS" if result.passed[i] else "FAIL"
            print(f"  {smi}: {score:.4f} (raw={raw}) [{status}]")
        print()


def _cmd_synthesis(config: LipConfig, args):
    from lip.scoring.synthesis import analyze_batch, is_available
    from pathlib import Path

    if not is_available():
        print("Error: AiZynthFinder is not installed.")
        print("Install with: pip install aizynthfinder")
        sys.exit(1)

    smiles_list = []
    if hasattr(args, "smiles") and args.smiles:
        smiles_list = args.smiles
    elif hasattr(args, "input_csv") and args.input_csv:
        from lip.utils.io import load_smiles_from_csv
        smiles_list = load_smiles_from_csv(args.input_csv,
                                           column=getattr(args, "smiles_column", "smiles"))

    if not smiles_list:
        print("Error: No SMILES provided. Use --smiles or --input-csv")
        sys.exit(1)

    print(f"Analyzing synthesis for {len(smiles_list)} molecule(s)...")

    results = analyze_batch(
        smiles_list,
        config_path=config.paths.aizynthfinder_config or None,
        time_limit=config.synthesis.time_limit,
    )

    n_solved = sum(1 for r in results if r.is_solved)
    print(f"\nResults: {n_solved}/{len(results)} solved")
    for r in results:
        status = "SOLVED" if r.is_solved else "UNSOLVED"
        print(f"  [{status}] {r.smiles}: {r.n_routes} routes, best_score={r.best_score:.3f}")
        if r.routes:
            best = r.routes[0]
            print(f"    Best route: {best.n_steps} steps, "
                  f"{len(best.starting_materials)} starting materials")

    if config.output_dir:
        from lip.utils.io import save_json
        from dataclasses import asdict
        output_path = Path(config.output_dir) / "synthesis_analysis.json"
        save_json([asdict(r) for r in results], output_path)
        print(f"\nResults saved to {output_path}")


def _cmd_pocket2mol(config: LipConfig, args):
    from lip.pocket2mol.generate import run_pocket2mol

    if not config.receptor_pdb:
        print("Error: --receptor is required for pocket2mol")
        sys.exit(1)

    sdf_paths = run_pocket2mol(
        pdb_path=config.receptor_pdb,
        pocket_center=config.pocket_center,
        n_samples=args.n_samples,
        bbox_size=args.bbox_size,
        pocket2mol_dir=config.paths.pocket2mol_dir,
    )

    print(f"Generated {len(sdf_paths)} molecule(s):")
    for p in sdf_paths:
        print(f"  {p}")


# ---------------------------------------------------------------------------
# Argument definitions
# ---------------------------------------------------------------------------

def _add_common_args(parser: argparse.ArgumentParser):
    parser.add_argument("--config", type=str, default=None, help="YAML config file")
    parser.add_argument("--receptor", dest="receptor_pdb", type=str, default=None)
    parser.add_argument("--pocket-center", dest="pocket_center", type=float, nargs=3, default=None)
    parser.add_argument("--output-dir", dest="output_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"])
    parser.add_argument("--scoring-method", dest="scoring_method", type=str, default=None,
                        choices=["weighted_sum", "pareto"])
    parser.add_argument("--is-docked", dest="is_docked", action="store_true", default=None,
                        help="PDB already contains docked ligand")
    parser.add_argument("--no-docked", dest="is_docked", action="store_false",
                        help="PDB and ligand SDF are separate (requires --ligand-sdf)")
    parser.add_argument("--ligand-sdf", dest="ligand_sdf", type=str, default=None,
                        help="Ligand SDF file path (required when --no-docked)")
    parser.add_argument("--log-level", dest="log_level", type=str, default="info",
                        choices=["debug", "info", "warning", "error"])


def _add_generator_args(parser: argparse.ArgumentParser):
    g = parser.add_argument_group("Generator")
    g.add_argument("--prior-model", dest="prior_model", type=str, default=None)
    g.add_argument("--agent-model", dest="agent_model", type=str, default=None)
    g.add_argument("--batch-size", dest="batch_size", type=int, default=None)
    g.add_argument("--sigma", type=int, default=None)
    g.add_argument("--learning-rate", dest="learning_rate", type=float, default=None)
    g.add_argument("--diversity-filter", dest="diversity_filter", type=str, default=None)


def _add_optimization_args(parser: argparse.ArgumentParser):
    g = parser.add_argument_group("Optimization")
    g.add_argument("--mode", type=str, default=None, choices=["managed", "manual"])
    g.add_argument("--n-steps", dest="n_steps", type=int, default=None)
    g.add_argument("--opt-batch-size", dest="opt_batch_size", type=int, default=None)
    g.add_argument("--max-score", dest="max_score", type=float, default=None)
    g.add_argument("--early-stop-patience", dest="early_stop_patience", type=int, default=None)
    g.add_argument("--inception-memory-size", dest="inception_memory_size", type=int, default=None)
    g.add_argument("--inception-sample-size", dest="inception_sample_size", type=int, default=None)
    g.add_argument("--inception-retention", dest="inception_retention", type=str, default=None,
                    choices=["top", "random"])
    g.add_argument("--n-molecules-per-round", dest="n_molecules_per_round", type=int, default=None)
    g.add_argument("--n-rounds", dest="n_rounds", type=int, default=None)
    g.add_argument("--diversity-threshold", dest="diversity_threshold", type=float, default=None)
    g.add_argument("--checkpoint-every", dest="checkpoint_every", type=int, default=None)


def _add_docking_args(parser: argparse.ArgumentParser):
    g = parser.add_argument_group("Docking")
    g.add_argument("--exhaustiveness", type=int, default=None)
    g.add_argument("--box-size", dest="box_size", type=int, default=None)
    g.add_argument("--docking-weight", dest="docking_weight", type=float, default=None)
    g.add_argument("--transform-high", dest="transform_high", type=float, default=None)
    g.add_argument("--transform-low", dest="transform_low", type=float, default=None)
    g.add_argument("--no-interactions", dest="analyze_interactions",
                    action="store_false", default=None)
    g.add_argument("--interaction-weight", dest="interaction_weight", type=float, default=None)


def _add_constraint_args(parser: argparse.ArgumentParser):
    g = parser.add_argument_group("Constraints")
    g.add_argument(
        "--constraint", action="append", default=None, dest="cli_constraints",
        help="Constraint spec: type:param1:param2:...:weight (repeatable)",
    )
    g.add_argument("--filter-lipinski", dest="filter_lipinski", type=_str_to_bool, default=None)
    g.add_argument("--filter-pains", dest="filter_pains", type=_str_to_bool, default=None)


def _add_synthesis_args(parser: argparse.ArgumentParser):
    g = parser.add_argument_group("Synthesis")
    g.add_argument("--no-synthesis", dest="synthesis_enabled",
                    action="store_false", default=None,
                    help="Disable post-optimization synthesis analysis")
    g.add_argument("--synthesis-top-n", dest="synthesis_top_n", type=int, default=None,
                    help="Number of top molecules to analyze for synthesis (default: 10)")
    g.add_argument("--synthesis-time-limit", dest="synthesis_time_limit", type=int, default=None,
                    help="MCTS time limit per molecule in seconds (default: 120)")


def _add_path_args(parser: argparse.ArgumentParser):
    g = parser.add_argument_group("External tool paths")
    g.add_argument("--pocket2mol-dir", dest="pocket2mol_dir", type=str, default=None)
    g.add_argument("--aizynthfinder-config", dest="aizynthfinder_config", type=str, default=None)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _str_to_bool(v: str) -> bool:
    return v.lower() in ("true", "1", "yes")


def _args_to_dict(args: argparse.Namespace) -> dict:
    """Convert argparse Namespace to dict, excluding None values and internals."""
    d = {}
    for k, v in vars(args).items():
        if v is not None and k not in ("command", "config", "log_level", "resume",
                                        "smiles", "cli_constraints",
                                        "n_samples", "bbox_size",
                                        "input_csv", "smiles_column"):
            d[k] = v

    # Parse CLI constraints
    if hasattr(args, "cli_constraints") and args.cli_constraints:
        from lip.config import ConstraintConfig
        constraints = _parse_cli_constraints(args.cli_constraints)
        if constraints:
            d["_cli_constraints"] = constraints

    return d


def _parse_cli_constraints(specs: list[str]) -> list:
    """Parse constraint specs from CLI.

    Formats:
        property_range:molecular_weight:200:500:0.1:1.0
        qed:1.0
        similarity:CCO:maximize:0.5
        smarts:[NH2]:true:1.0
    """
    from lip.config import ConstraintConfig

    constraints = []
    for spec in specs:
        parts = spec.split(":")
        ctype = parts[0]

        if ctype == "property_range" and len(parts) >= 5:
            constraints.append(ConstraintConfig(
                type="property_range",
                weight=float(parts[5]) if len(parts) > 5 else 1.0,
                params={
                    "property": parts[1],
                    "min": float(parts[2]),
                    "max": float(parts[3]),
                    "margin": float(parts[4]),
                },
            ))
        elif ctype == "qed":
            constraints.append(ConstraintConfig(
                type="qed",
                weight=float(parts[1]) if len(parts) > 1 else 1.0,
            ))
        elif ctype == "sa":
            constraints.append(ConstraintConfig(
                type="sa",
                weight=float(parts[1]) if len(parts) > 1 else 1.0,
            ))
        elif ctype == "similarity" and len(parts) >= 2:
            constraints.append(ConstraintConfig(
                type="similarity",
                weight=float(parts[3]) if len(parts) > 3 else 1.0,
                params={
                    "reference_smiles": parts[1],
                    "mode": parts[2] if len(parts) > 2 else "maximize",
                },
            ))
        elif ctype == "smarts" and len(parts) >= 2:
            constraints.append(ConstraintConfig(
                type="smarts",
                weight=float(parts[3]) if len(parts) > 3 else 1.0,
                params={
                    "pattern": parts[1],
                    "must_match": parts[2].lower() != "false" if len(parts) > 2 else True,
                },
            ))
        else:
            logging.getLogger(__name__).warning(f"Unknown constraint spec: {spec}")

    return constraints


if __name__ == "__main__":
    main()
