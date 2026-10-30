"""Planned predict entry point. No model implementation is released yet."""
import argparse

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Reviewed TOML experiment specification")
    parser.parse_args()
    parser.exit(2, "OcuWeave predict is not implemented. This release is a research scaffold.\n")

if __name__ == "__main__": main()
