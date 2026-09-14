import sys
import json
import argparse
from pathlib import Path
import pandas as pd
from .evaluation import run
from .plots import plot_results


def explain(args):
    """Block-level explanation of the fitted models for one or more targets."""
    from . import explain as ex
    from .selection import candidate_groups
    table = pd.read_csv(args.features, dtype={'participant_id': str, 'group_id': str,
                                              'release': str, 'sex': str})
    config = json.loads(Path(args.config).read_text())
    duration = config.get('duration', 60)
    groups = candidate_groups(table.columns, config.get('candidates', 'blocks'))
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    solo, permuted = {}, {}
    for target in args.targets:
        frame = apply_target(table, target, duration)
        frame = frame[frame.duration == duration].reset_index(drop=True)
        s = ex.block_solo_performance(frame, groups, config)
        p = ex.block_importance(frame, groups, config, n_repeats=args.repeats)
        s.to_csv(out / f'{target}_block_solo.csv', index=False)
        p.to_csv(out / f'{target}_block_importance.csv', index=False)
        solo[target], permuted[target] = s, p
        print(f'{target}: best single block '
              f'{s.groupby("block").balanced_accuracy.mean().idxmax()} '
              f'({s.groupby("block").balanced_accuracy.mean().max():.3f})', flush=True)
    ex.plot_solo_heatmap(solo, out / 'solo_blocks', 'Single-block held-out accuracy')
    ex.plot_block_importance(permuted, out / 'block_importance', 'Block permutation importance')
    print(f'Wrote explanation tables and figures to {out}')


def apply_target(table, target, duration):
    """Derive a binary label. Controls exist to make a null interpretable."""
    t = table.copy()
    if target == 'label':
        return t
    if target == 'age_band':
        t['label'] = (t.age >= t[t.duration == duration].age.median()).astype(int)
        return t
    if target == 'sex':
        t['label'] = (t.sex == 'F').astype(int)
        return t
    raise ValueError("target must be label, age_band or sex")


def main():
    if len(sys.argv) > 1 and sys.argv[1] not in ('search', 'plot', 'explain', '-h', '--help'):
        from hbn_r3.cli import main as data_main
        return data_main()
    parser = argparse.ArgumentParser(
        description='SeqMap: search, plot and explain, plus the shared inspect-data, '
                    'inspect-bdf, prepare-cohort, extract and demo commands')
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('search')
    p.add_argument('--features', required=True)
    p.add_argument('--config', required=True)
    p.add_argument('--output', required=True)
    p = sub.add_parser('plot')
    p.add_argument('--results', required=True)
    p = sub.add_parser('explain')
    p.add_argument('--features', required=True)
    p.add_argument('--config', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--targets', nargs='+', default=['label', 'sex', 'age_band'],
                   choices=['label', 'sex', 'age_band'])
    p.add_argument('--repeats', type=int, default=30)
    args = parser.parse_args()
    if args.command == 'plot':
        plot_results(args.results)
    elif args.command == 'explain':
        explain(args)
    else:
        table = pd.read_csv(args.features, dtype={'participant_id': str, 'group_id': str,
                                                  'release': str, 'sex': str})
        run(table, json.loads(Path(args.config).read_text()), args.output)


if __name__ == '__main__':
    main()
