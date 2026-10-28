from pathlib import Path
import argparse
import csv
import gzip
import json
from train import build_manifest


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Prepare original test sites for official m6Anet inference")
    parser.add_argument('--data0', type=Path, required=True)
    parser.add_argument('--data1', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, default=root / 'splits/gene_split.csv')
    parser.add_argument('--labels0', type=Path, default=root.parent / 'data0/data.info.labelled')
    parser.add_argument('--labels1', type=Path, default=root.parent / 'data1/data.info')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    # Avoid mixing newly prepared inputs with files from an earlier run.
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError('Choose a new m6Anet input directory')
    MANIFEST_PATH = args.manifest
    OUTPUT_DIR = args.output_dir
    DATA_PATHS = {'data0': args.data0, 'data1': args.data1}
    # Load the exact held-out sites used in the previous experiment.
    wanted = {name: set() for name in DATA_PATHS}

    manifest = build_manifest(MANIFEST_PATH, args.labels0, args.labels1)
    for row in manifest.to_dict('records'):
        source = row["source_dataset"]
        if row["split"] == "test" and source in wanted:
            wanted[source].add(
                (
                    row["transcript_id"],
                    int(row["transcript_position"]),
                )
            )

    # Create a separate m6Anet input directory for each source dataset.
    for source, input_path in DATA_PATHS.items():
        output_path = OUTPUT_DIR / source
        output_path.mkdir(parents=True, exist_ok=True)

        found = set()
        eligible_count = 0
        low_coverage_count = 0

        opener = gzip.open if str(input_path).endswith('.gz') else open
        with opener(input_path, "rt", encoding="utf-8") as input_file, \
            (output_path / "data.json").open("wb") as json_file, \
            (output_path / "data.info").open("w", newline="") as info_file:
            
            writer = csv.writer(info_file)
            writer.writerow([
                "transcript_id",
                "transcript_position",
                "n_reads",
                "start",
                "end",
            ])

            for line in input_file:
                if not line.strip():
                    continue

                record = json.loads(line)

                for transcript_id, positions in record.items():
                    for position, sequences in positions.items():
                        key = (transcript_id, int(position))

                        # Only prepare test sites from the shared gene split.
                        if key not in wanted[source]:
                            continue

                        if key in found:
                            raise ValueError(f"Duplicate site: {source}:{key}")
                        # Track presence before filtering so low coverage is not counted as missing.
                        found.add(key)

                        if len(sequences) != 1:
                            raise ValueError(f"Unexpected sequences: {key}")

                        sequence, reads = next(iter(sequences.items()))

                        if not reads or any(len(read) != 9 for read in reads):
                            raise ValueError(f"Unexpected read features: {key}")

                        # m6Anet requires at least 20 reads per site.
                        if len(reads) < 20:
                            low_coverage_count += 1
                            continue

                        # Append synthetic local read IDs for output tracking.
                        indexed_reads = [
                            read + [index]
                            for index, read in enumerate(reads)
                        ]

                        output_record = {
                            transcript_id: {
                                str(int(position)): {
                                    sequence: indexed_reads
                                }
                            }
                        }

                        # Write ASCII bytes so file offsets match the positions in data.info.
                        encoded = (
                            json.dumps(
                                output_record,
                                separators=(",", ":"),
                                ensure_ascii=True,
                            )
                            + "\n"
                        ).encode("ascii")

                        # Compute offsets from the newly written JSON file.
                        start = json_file.tell()
                        json_file.write(encoded)
                        end = json_file.tell()

                        writer.writerow([
                            transcript_id,
                            int(position),
                            len(reads),
                            start,
                            end,
                        ])
                        eligible_count += 1

        # Check that every requested test site was present in the source file.
        missing = wanted[source] - found
        if missing:
            raise ValueError(
                f"{source}: {len(missing)} test sites were not found"
            )

        print(f"\n{source}")
        print(f"Requested test sites: {len(wanted[source]):,}")
        print(f"Prepared sites: {eligible_count:,}")
        print(f"Excluded sites with fewer than 20 reads: {low_coverage_count:,}")
        print(f"Input directory: {output_path}")


if __name__ == "__main__":
    main()
