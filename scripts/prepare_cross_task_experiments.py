"""Write the prespecified cross-task design matrix without starting training."""
import argparse
from ocuweave.experiments.roadmap import write_plan

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    plan = write_plan(args.output)
    print(f"Design only: {plan['core_runs']} core + {plan['conditional_runs']} conditional; 0 completed")
