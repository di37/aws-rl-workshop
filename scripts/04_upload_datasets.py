"""Upload the prompt datasets to S3 as Parquet (skips unchanged files).

64 training tickets and 32 held-out evaluation tickets, validated (at least 32
unique prompts each) and converted to the Parquet files the MTRL trainer and
evaluator read. A file whose bytes match the S3 object is not uploaded again.
"""

# region Imports & setup
from __future__ import annotations

import _bootstrap as bootstrap

# endregion


# region Entry point
def main() -> None:
    """Uploads changed dataset files and prints their S3 locations."""
    bootstrap.parse_args(__doc__.splitlines()[0])
    bootstrap.banner("Upload datasets (04)", "data/*.csv -> S3 Parquet")
    uploads = bootstrap.load_workflow().upload_datasets()
    bootstrap.show([{"split": split, "uri": info["uri"],
                     "action": "uploaded" if info["uploaded"] else "unchanged, kept"}
                    for split, info in uploads.items()])


if __name__ == "__main__":
    bootstrap.run(main)
# endregion
