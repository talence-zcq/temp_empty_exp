import argparse

def parse():
    p = argparse.ArgumentParser("Auto Pooling parse", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--feature_noise' ,type=int, default=1, help="dataset noise")
    p.add_argument('--hidden_dim', type=int, default=32, help='number of hidden features')
    p.add_argument('--hyper_data', type=str, default='ModelNet40', help='[20newsW100、ModelNet40、walmart-trips-100]')
    p.add_argument('--learning_rate', type=float, default=1e-2, help='')
    p.add_argument('--normal_data', type=str, default='MUTAG', help='[MUTAG]')
    p.add_argument('--seed', type=int, default=42, help='seed for randomness')
    p.add_argument('--schedule_factor', type=float, default=0.1, help='')
    p.add_argument('--schedule_patience', type=int, default=10, help='')
    p.add_argument('--weight_decay', type=float, default=1e-5,help='')


    return p.parse_args()