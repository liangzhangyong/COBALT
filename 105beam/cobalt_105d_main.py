"""Paper-aligned COBALT entry point for the 105-beam benchmark."""

import sys

from run_cobalt import main


if __name__ == "__main__":
    if "--benchmark" not in sys.argv:
        sys.argv[1:1] = ["--benchmark", "105beam"]
    main()
