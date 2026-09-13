import sys
import json
import argparse
from pathlib import Path
import pandas as pd
from .evaluation import run
from .plots import plot_results


def main():
    if len(sys.argv)>1 and sys.argv[1] not in ('search','plot','-h','--help'):
        from hbn_r3.cli import main as data_main
        return data_main()
    parser=argparse.ArgumentParser(description='SeqMap: use search/plot, or shared inspect-data, inspect-bdf, prepare-cohort, extract, demo commands')
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('search');p.add_argument('--features',required=True);p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    p=sub.add_parser('plot');p.add_argument('--results',required=True)
    args=parser.parse_args()
    if args.command=='plot':plot_results(args.results)
    else:
        table=pd.read_csv(args.features,dtype={'participant_id':str,'group_id':str,'release':str,'sex':str})
        run(table,json.loads(Path(args.config).read_text()),args.output)

if __name__=='__main__':main()
