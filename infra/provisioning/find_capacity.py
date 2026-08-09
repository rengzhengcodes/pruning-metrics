"""Find a fulfillable EC2 GPU spot capacity for the Qwen2-72B run.

Spot capacity for ``p5.48xlarge`` / ``p4de.24xlarge`` / ``p4d.24xlarge`` is
sporadic — even when the account has the vCPU quota, a particular AZ may have
no idle hosts at any given moment. This script walks a priority list of
``(region, instance_type)`` pairs, asks EC2 for the most recent spot price
per AZ, filters out AZs that do not currently advertise the instance type,
and prints the cheapest option as JSON.

The output is consumed by ``infra/provisioning/launch_gpu_instance.py`` (and a
launcher subagent) which then issues ``RunInstances`` with the corresponding
``InstanceMarketOptions`` block.

Example
-------
::

    AWS_PROFILE=rengz python infra/provisioning/find_capacity.py \\
        --regions us-east-1,us-west-2,us-east-2 \\
        --instance-types p5.48xlarge,p4de.24xlarge,p4d.24xlarge

The exit code is ``0`` when at least one candidate is found, ``2`` otherwise.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import boto3

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# pylint: disable=wrong-import-position
from infra.runners._runner_common import (  # noqa: E402
    boto_client_config,
    split_csv,
)


def parse_args() -> argparse.Namespace:
    """Parse capacity-finder arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Probe EC2 spot capacity across regions/AZs/instance types and "
            "print the cheapest currently-available choice as JSON."
        )
    )
    parser.add_argument(
        "--regions",
        default=os.environ.get("EC2_REGION_PRIORITY", "us-east-1,us-west-2,us-east-2"),
        help="Comma-separated AWS regions to probe (priority order).",
    )
    parser.add_argument(
        "--instance-types",
        default=os.environ.get(
            "EC2_INSTANCE_TYPE_PRIORITY",
            "p5.48xlarge,p4de.24xlarge,p4d.24xlarge",
        ),
        help="Comma-separated instance types to consider (priority order).",
    )
    parser.add_argument(
        "--lookback-hours",
        type=int,
        default=2,
        help="Spot-price-history lookback window for currentness check.",
    )
    parser.add_argument(
        "--max-price-multiplier",
        type=float,
        default=2.5,
        help=(
            "Cap on the spot bid expressed as a multiplier of the most recent "
            "observed spot price. Defaults to 2.5x to handle transient price "
            "spikes while running the experiment."
        ),
    )
    return parser.parse_args()


def offered_azs_by_type(ec2: Any, instance_types: list[str]) -> dict[str, set[str]]:
    """Return the AZs that *currently* offer each instance type.

    One paginated ``DescribeInstanceTypeOfferings`` sweep covers every
    instance type (the filter accepts a list), instead of one call per type.
    """

    paginator = ec2.get_paginator("describe_instance_type_offerings")
    offered: dict[str, set[str]] = {t: set() for t in instance_types}
    for page in paginator.paginate(
        LocationType="availability-zone",
        Filters=[{"Name": "instance-type", "Values": instance_types}],
    ):
        for offering in page.get("InstanceTypeOfferings", []):
            offered[offering["InstanceType"]].add(offering["Location"])
    return offered


def latest_spot_prices(
    ec2: Any,
    instance_types: list[str],
    lookback_hours: int,
) -> dict[tuple[str, str], dict[str, Any]]:
    """Return the most recent spot price per ``(instance_type, AZ)``.

    One paginated ``DescribeSpotPriceHistory`` sweep covers every instance
    type, instead of one call per type.
    """

    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=lookback_hours)
    paginator = ec2.get_paginator("describe_spot_price_history")
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for page in paginator.paginate(
        InstanceTypes=instance_types,
        ProductDescriptions=["Linux/UNIX"],
        StartTime=start,
        EndTime=end,
    ):
        for entry in page.get("SpotPriceHistory", []):
            key = (entry["InstanceType"], entry["AvailabilityZone"])
            timestamp = entry["Timestamp"]
            if key not in latest or timestamp > latest[key]["timestamp"]:
                latest[key] = {
                    "timestamp": timestamp,
                    "price": float(entry["SpotPrice"]),
                }
    return latest


def find_candidates(
    regions: list[str],
    instance_types: list[str],
    lookback_hours: int,
    max_price_multiplier: float,
) -> list[dict[str, Any]]:
    """Return all viable ``(region, instance_type, az, price)`` candidates."""

    candidates: list[dict[str, Any]] = []
    # Outer loop over regions then instance types preserves priority semantics
    # for ties, but we do not stop early so the caller can see fallbacks.
    for region_index, region in enumerate(regions):
        # One client per region, reused by both describe sweeps.
        ec2 = boto3.client("ec2", region_name=region, config=boto_client_config())
        try:
            offered = offered_azs_by_type(ec2, instance_types)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            print(
                f"WARN: describe_instance_type_offerings failed in {region}: {exc}",
                file=sys.stderr,
            )
            continue
        if not any(offered.values()):
            continue
        try:
            prices = latest_spot_prices(ec2, instance_types, lookback_hours)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            print(
                f"WARN: describe_spot_price_history failed in {region}: {exc}",
                file=sys.stderr,
            )
            continue
        for type_index, instance_type in enumerate(instance_types):
            for az in offered[instance_type]:
                price_entry = prices.get((instance_type, az))
                if price_entry is None:
                    continue
                spot_price = price_entry["price"]
                candidates.append(
                    {
                        "region": region,
                        "instance_type": instance_type,
                        "availability_zone": az,
                        "spot_price_usd_per_hour": spot_price,
                        "max_bid_usd_per_hour": round(
                            spot_price * max_price_multiplier, 4
                        ),
                        "observed_at_utc": price_entry["timestamp"].isoformat(),
                        "priority_region_index": region_index,
                        "priority_instance_index": type_index,
                    }
                )
    # Sort: keep configured priority first; tie-break on price.
    candidates.sort(
        key=lambda candidate: (
            candidate["priority_region_index"],
            candidate["priority_instance_index"],
            candidate["spot_price_usd_per_hour"],
        )
    )
    return candidates


def main() -> int:
    """CLI entry point. Prints JSON of viable candidates."""

    args = parse_args()
    regions = list(split_csv(args.regions) or ())
    instance_types = list(split_csv(args.instance_types) or ())
    candidates = find_candidates(
        regions=regions,
        instance_types=instance_types,
        lookback_hours=args.lookback_hours,
        max_price_multiplier=args.max_price_multiplier,
    )
    print(
        json.dumps(
            {
                "regions": regions,
                "instance_types": instance_types,
                "num_candidates": len(candidates),
                "candidates": candidates,
            },
            indent=2,
            default=str,
        )
    )
    return 0 if candidates else 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
