"""Run from this directory: python3 check_contract.py. No model/GPU imports."""
from pathlib import Path
from contract import Report, TensorSpec, load_task


def main():
    task, digest = load_task(Path(__file__).with_name("task.json"))
    for rows in task["rows"]:
        for width in task["widths"]:
            TensorSpec(rows, width).validate()
    print(Report(task["task_id"], digest, None, "cpu", "contract_validated",
                 ("manifest_load", "positive_contiguous_bf16_shapes")).to_json())


if __name__ == "__main__":
    main()
