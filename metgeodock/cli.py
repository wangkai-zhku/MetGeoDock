"""Command-line interface for MetGeoDock."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from . import __version__
from .dock import dock_ligand, redock_rmsd
from .io import load_molecule, merge_with_waters, read_pdb, write_pdb
from .prepare import prepare_receptor
from .ranking import take_top, write_rank_table
from .screen import screen_library


def _cmd_dock(args: argparse.Namespace) -> int:
    protein = read_pdb(args.protein)
    if args.water:
        protein = merge_with_waters(protein, read_pdb(args.water))
    ligand = load_molecule(args.ligand)
    max_poses = int(args.max_poses)
    if args.top_n and args.top_n > 0:
        max_poses = max(max_poses, int(args.top_n))
    result = dock_ligand(
        protein,
        ligand,
        metal=args.metal,
        n_spin=args.n_spin,
        max_poses=max_poses,
        keep_water=bool(args.keep_water),
        multi_metal=bool(args.multi_metal),
        semi_flex=bool(args.semi_flex),
        rerank=str(args.rerank),
        model_path=args.model,
        coord_water=int(args.coord_water),
        donor_chem_scale=float(getattr(args, "donor_chem_scale", 0.0)),
        thermo=bool(getattr(args, "thermo", False)),
        include_chem=bool(getattr(args, "include_chem", True)),
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for pose in result.poses:
        rows.append(
            {
                "score": pose.score.terms.get("rank_score", pose.score.total),
                "metalgeo": pose.score.total,
                "geo_rmsd_or": pose.score.geo,
                "gp_nll": pose.score.gp,
                "clash": pose.score.clash,
                "dist_harm": pose.score.dist,
                "cn_pen": pose.score.cn,
                "map_overlap": pose.score.terms.get("map_overlap", 0.0),
                "n_water_displaced": pose.score.terms.get("n_water_displaced", 0.0),
                "motif": pose.motif,
                "n_coord_lig": pose.score.n_coord_lig,
                "n_water": pose.score.terms.get("n_water", result.sphere.n_water),
                "n_coord_water": pose.score.terms.get("n_coord_water", 0.0),
                "n_other_metal": pose.score.terms.get("n_other_metal", result.sphere.n_other_metal),
                "dock_metal_index": pose.score.terms.get("dock_metal_index", result.sphere.metal_index),
                "rerank": pose.score.terms.get("rerank", args.rerank),
                "ligand": pose.ligand,
            }
        )
    ranked = take_top(pd.DataFrame(rows), args.top_n)
    if args.write_poses:
        pose_dir = out / "poses"
        pose_dir.mkdir(parents=True, exist_ok=True)
        pose_paths = []
        for rec in ranked.to_dict(orient="records"):
            lig = rec.get("ligand")
            if lig is None:
                pose_paths.append("")
                continue
            dest = pose_dir / f"pose_{int(rec['rank']):04d}.pdb"
            write_pdb(lig, dest)
            pose_paths.append(str(dest))
        ranked = ranked.assign(pose=pose_paths)
    table = ranked.drop(columns=["ligand"], errors="ignore")
    write_rank_table(table, out / "ranks.csv")
    write_rank_table(table, out / "ranks.json")
    n_out = len(table)
    if args.reference and result.best:
        ref = load_molecule(args.reference)
        rmsd = redock_rmsd(result.best.ligand, ref)
        print(f"best_score={result.best.score.total:.4f} redock_rmsd_angstrom={rmsd:.3f} n_ranked={n_out}")
    elif result.best:
        print(f"best_score={result.best.score.total:.4f} n_ranked={n_out} wrote={out / 'ranks.csv'}")
    else:
        print("no poses generated", file=sys.stderr)
        return 1
    return 0


def _cmd_screen(args: argparse.Namespace) -> int:
    protein = read_pdb(args.protein)
    if args.water:
        protein = merge_with_waters(protein, read_pdb(args.water))
    frame = screen_library(
        protein,
        args.library,
        metal=args.metal,
        scorer=args.scorer,
        model_path=args.model,
        n_spin=args.n_spin,
        max_poses=args.max_poses,
        out_dir=args.pose_dir,
        top_n=args.top_n,
        write_poses=bool(args.write_poses or args.pose_dir),
        keep_water=bool(args.keep_water),
        multi_metal=bool(args.multi_metal),
        semi_flex=bool(args.semi_flex),
        rerank=str(args.rerank),
        coord_water=int(args.coord_water),
        donor_chem_scale=float(getattr(args, "donor_chem_scale", 0.0)),
        thermo=bool(getattr(args, "thermo", False)),
        include_chem=bool(getattr(args, "include_chem", True)),
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    print(f"wrote {len(frame)} rows to {out}")
    return 0



def _cmd_prepare(args: argparse.Namespace) -> int:
    info = prepare_receptor(
        args.pdb,
        out=args.out,
        ph=float(args.ph),
        metal=args.metal,
        metal_h_cutoff=float(args.metal_h_cutoff),
        keep_waters=bool(args.keep_water),
        water_cutoff=float(args.water_cutoff),
        report=args.report,
    )
    print(f"wrote {info['output']}")
    print(f"  atoms={info['n_atoms_out']} H={info['n_hydrogens']} "
          f"metals={info['n_metals']} waters={info['n_waters_kept']}")
    print(f"  missing atoms added={info['missing_atoms_added']} "
          f"(loops not rebuilt: {info['missing_residues_not_rebuilt']}), "
          f"H dropped near metal={info['hydrogens_removed_near_metal']}")
    for site in info["metal_sites"]:
        print(f"  {site['resname']} chain {site['chain']}: {site['n_coord']} coord -> "
              f"{', '.join(site['coordination'][:6])}")
    for w in info["warnings"]:
        print(f"  WARNING: {w}")
    return 0


def _cmd_score(args: argparse.Namespace) -> int:
    from .coordination import detect_coordination
    from .scoring import score_pose

    protein = read_pdb(args.protein)
    ligand = load_molecule(args.ligand)
    sphere = detect_coordination(protein, metal=args.metal)
    sc = score_pose(protein, ligand, sphere)
    print(json.dumps({"total": sc.total, **{k: v for k, v in sc.terms.items() if not isinstance(v, str)}}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="metgeodock",
        description="Geometry-matching virtual screening for metalloproteins.",
    )
    p.add_argument("--version", action="version", version=f"MetGeoDock {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("dock", help="dock one ligand onto a metalloprotein")
    d.add_argument("--protein", required=True)
    d.add_argument("--ligand", required=True)
    d.add_argument("--metal", default=None)
    d.add_argument("--out", default="processed/dock")
    d.add_argument("--n-spin", dest="n_spin", type=int, default=12)
    d.add_argument("--max-poses", dest="max_poses", type=int, default=100)
    d.add_argument(
        "--top-n",
        dest="top_n",
        type=int,
        default=0,
        help="keep this many ranked poses (100 or 1000). Default 0 = all scored poses, best to worst",
    )
    d.add_argument("--write-poses", dest="write_poses", action="store_true", help="also write PDB docking modes")
    d.add_argument("--reference", default=None, help="crystal ligand for RMSD")
    d.add_argument("--water", default=None, help="optional PDB of crystallographic waters to merge")
    d.add_argument(
        "--keep-water",
        dest="keep_water",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="if set, use up to --coord-water closest crystal first-shell O atoms as occupancy hypotheses. Default: strip pocket HOH and place dummy coordinating waters on vacant vertices",
    )
    d.add_argument(
        "--coord-water",
        dest="coord_water",
        type=int,
        default=2,
        help="sample 0..N coordinating waters on vacant metal sites (default 2). Changes ligand denticity; not bulk pocket water",
    )
    d.add_argument(
        "--multi-metal",
        dest="multi_metal",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="dock every pocket metal and merge poses (default). --no-multi-metal uses the first ion",
    )
    d.add_argument(
        "--semi-flex",
        dest="semi_flex",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="rotate donor-adjacent torsions on top seeds if map overlap does not fall",
    )
    d.add_argument(
        "--rerank",
        dest="rerank",
        choices=("metgeo", "posescore", "posescore-nn", "posenet", "ranker", "ranker-blend"),
        default="metgeo",
        help="pose ranker. ranker = trained MetGeoDock listwise net; posescore = analytic",
    )
    d.add_argument("--model", default=None, help="optional PoseScore/PoseNet checkpoint")
    d.set_defaults(func=_cmd_dock)

    s = sub.add_parser("screen", help="batch virtual screening")
    s.add_argument("--protein", required=True)
    s.add_argument("--library", required=True, help="MOL2 file or directory")
    s.add_argument("--metal", default=None)
    s.add_argument("--scorer", choices=("hybrid", "metgeo", "metalgeo", "ml", "coordnet"), default="metgeo")
    s.add_argument("--model", default=None)
    s.add_argument("--out", default="results/screen.csv")
    s.add_argument(
        "--top-n",
        dest="top_n",
        type=int,
        default=0,
        help="keep this many ranked ligands (100 or 1000). Default 0 = all scored molecules, best to worst",
    )
    s.add_argument("--write-poses", dest="write_poses", action="store_true", help="write best PDB per ranked hit")
    s.add_argument("--pose-dir", dest="pose_dir", default=None)
    s.add_argument(
        "--no-chem",
        dest="include_chem",
        action="store_false",
        help="score with the geometric terms only; the default adds the hydrophobic, "
             "rotation and desolvation terms used in the paper",
    )
    s.add_argument("--n-spin", dest="n_spin", type=int, default=8)
    s.add_argument("--max-poses", dest="max_poses", type=int, default=5)
    s.add_argument("--water", default=None, help="optional PDB of crystallographic waters to merge")
    s.add_argument(
        "--keep-water",
        dest="keep_water",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    s.add_argument("--coord-water", dest="coord_water", type=int, default=2)
    s.add_argument(
        "--multi-metal",
        dest="multi_metal",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    s.add_argument(
        "--semi-flex",
        dest="semi_flex",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    s.add_argument(
        "--rerank",
        dest="rerank",
        choices=("metgeo", "posescore", "posescore-nn", "posenet", "ranker", "ranker-blend"),
        default="metgeo",
    )
    s.add_argument(
        "--donor-chem-scale",
        dest="donor_chem_scale",
        type=float,
        default=0.0,
        help=">0 enables metal-specific donor-group penalties (-ln strength * scale); "
             "0 (default) keeps the original purely geometric scoring",
    )
    s.add_argument(
        "--thermo",
        dest="thermo",
        action="store_true",
        default=True,
        help="DEFAULT ON. Adds the literature-constant thermodynamic terms to the ranking "
             "score: hydrophobic burial (-0.025 kcal/mol/A^2) and conformational adaptability "
             "(-0.6 kcal/mol per rotatable bond, coupled to buried area). No fitted parameter; "
             "costs ~0.22 s per ligand. Use --no-thermo for the purely geometric score.",
    )
    s.add_argument(
        "--no-thermo",
        dest="thermo",
        action="store_false",
        help="reproduce the purely geometric MetGeo score (pre-thermo behaviour).",
    )
    s.set_defaults(func=_cmd_screen)

    pr = sub.add_parser(
        "prepare",
        help="add hydrogens + report metal sites: build a docking-ready all-atom receptor",
    )
    pr.add_argument("--pdb", required=True, help="input PDB (heavy atoms, crystal coordinates)")
    pr.add_argument("--out", default=None, help="output PDB (default: <stem>_h.pdb)")
    pr.add_argument("--ph", type=float, default=7.4, help="pH for protonation states")
    pr.add_argument("--metal", default=None, help="metal hint used only in the report, e.g. zn")
    pr.add_argument("--metal-h-cutoff", dest="metal_h_cutoff", type=float, default=2.6,
                    help="hydrogens closer than this to a metal are removed")
    pr.add_argument("--keep-water", dest="keep_water", action=argparse.BooleanOptionalAction,
                    default=True, help="keep crystal waters near the metal site")
    pr.add_argument("--water-cutoff", dest="water_cutoff", type=float, default=10.0)
    pr.add_argument("--report", default=None, help="write a JSON report here")
    pr.set_defaults(func=_cmd_prepare)

    sc = sub.add_parser("score", help="score an existing pose")
    sc.add_argument("--protein", required=True)
    sc.add_argument("--ligand", required=True)
    sc.add_argument("--metal", default=None)
    sc.set_defaults(func=_cmd_score)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
