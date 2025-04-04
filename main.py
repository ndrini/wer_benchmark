import sys

from asr import compute_asr_benchmark


def display_info():
    """Display information about the ASR engines and their capabilities."""
    engines: list = compute_asr_benchmark.get_asr_info()
    print(f"Available ASR engines: {engines}")


def manage_starting_info(args: list[str] = []):
    """Provide info about the possible model or the start the execution with the
    specified number of cases and the specified engines (or all the engines)"""

    if not args:
        print("Wrong command")
        print(
            "You must follow the template: python -m main <info|N> [all|model1,model2,...]"
        )
        sys.exit(0)

    if args[0] == "info":
        display_info()
        sys.exit(0)

    try:
        item_number = int(args[0])
    except ValueError:
        print("Error: first argument MUST be either 'info' or a int number.")
        sys.exit(1)

    if len(args) > 1:
        if args[1] == "all":
            engines: list = compute_asr_benchmark.get_asr_info()
            wer_result = compute_asr_benchmark.get_wer(item_number, engines)
        else:

            engines = [engine.strip(",") for engine in args[1:]]
            print(f"List of available engines: {engines}")

            # check all engines are supported
            for engine in engines:
                if engine not in compute_asr_benchmark.get_asr_info():
                    print(f"Error: {engine} is not a supported ASR engine.")
                    sys.exit(1)
            wer_result = compute_asr_benchmark.get_wer(item_number, engines)

        print(wer_result)


if __name__ == "__main__":
    args = sys.argv[1:]  # read input from the command line
    manage_starting_info(args)
