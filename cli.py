import argparse
import ppoi.main

def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--initialize', action='store_true')
    parser.add_argument('--learn', action='store_true')
    parser.add_argument('--describe', action='store_true')
    parser.add_argument('--interactive', action='store_true')
    parser.add_argument('--find', action='store_true')
    args = parser.parse_args()

    if args.initialize:
        ppoi.main._initialize()

    if args.learn:
        ppoi.main._learn()

    if args.describe:
        ppoi.main._learn()
        ppoi.main._describe()

    if args.interactive:
        ppoi.main._interactive()

    if args.find:
        ppoi.main._find()


if __name__ == "__main__":
    _main()
