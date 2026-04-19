from __future__ import annotations

import argparse
import sys

from billwise.common.doctor import collect_doctor_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 1 if any failing checks are found",
    )
    args = parser.parse_args()

    report = collect_doctor_report()

    print(f"BillWise Doctor")
    print(f"Project root: {report['project_root']}")
    print("-" * 80)

    for check in report["checks"]:
        status = check["status"].upper().ljust(4)
        name = check["name"].ljust(40)
        print(f"[{status}] {name} {check['detail']}")

    print("-" * 80)
    print(
        f"Summary: ok={report['summary']['ok']}  "
        f"warn={report['summary']['warn']}  "
        f"fail={report['summary']['fail']}"
    )

    if args.strict and report["summary"]["fail"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()