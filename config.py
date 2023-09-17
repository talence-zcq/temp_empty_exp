import argparse

def parse():
    p = argparse.ArgumentParser("Auto Pooling parse", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--data', type=str, default='walmart-trips-100' , help='[20newsW100、ModelNet40、walmart-trips-100]')
    p.add_argument('--feature_noise' ,type=int, default=1, help="dataset noise")
    p.add_argument('--seed', type=int, default=42, help='seed for randomness')

    return p.parse_args()