import argparse

parser = argparse.ArgumentParser(description='Robust Temporal Difference Learning')
    
# Arguments
parser.add_argument('--graph', type=str, default='CompleteGraph')
parser.add_argument('--aggregation', type=str, default='mean')
parser.add_argument('--attack', type=str, default='none')
parser.add_argument('--data-partition', type=str, default='iid')
parser.add_argument('--lr-ctrl', type=str, default='1/sqrt k')

parser.add_argument('--no-fixed-seed', action='store_true',
                    help="If specifed, the random seed won't be fixed")
parser.add_argument('--seed', type=int, default=100)

parser.add_argument('--without-record', action='store_true',
                    help='If specifed, no file of running record and log will be left')
parser.add_argument('--step-agg', type=int, default=1)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--prob', type=float, default=1.0)

args = parser.parse_args()
gpu = args.gpu
