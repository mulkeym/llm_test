import argparse
from .runner import BenchmarkRunner


def main():
    parser = argparse.ArgumentParser(description="LLM Benchmark Suite")
    parser.add_argument("--model", required=True, help="Model name for this run")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--export", choices=["json", "csv"], help="Export results after run")
    args = parser.parse_args()

    runner = BenchmarkRunner(config_path=args.config)
    runner.run(model_name=args.model, export_format=args.export)


if __name__ == "__main__":
    main()
