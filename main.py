"""Master thesis experiment entry point.

The default configuration trains depth estimation on RS3DBench:

    python main.py --config configs/rs3dbench_depth.yaml

The previous EuroSAT classification configurations remain available.
"""

import sys

from scripts.run_experiments import main


if __name__ == "__main__":
    if "--config" not in sys.argv:
        sys.argv.extend(["--config", "configs/rs3dbench_depth.yaml"])
    main()
