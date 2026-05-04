"""Find a fulfillable EC2 GPU spot capacity for the Qwen2-72B run.

Spot capacity for ``p5.48xlarge`` / ``p4de.24xlarge`` / ``p4d.24xlarge`` is
sporadic — even when the account has the vCPU quota, a particular AZ may have
no idle hosts at any given moment. This script walks a priority list of
``(region, instance_type)`` pairs, asks EC2 for the most recent spot price
per AZ, filters out AZs that do not currently advertise the instance type,
and prints the cheapest option as JSON.

The output is consumed by ``infra/ec2/launch_gpu_instance.py`` (and a
launcher subagent) which then issues ``RunInstances`` with the corresponding
``InstanceMarketOptions`` block.

Example
-------
::

    AWS_PROFILE=rengz python infra/ec2/find_capacity.py \\
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
from typing import Any

import boto3


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
        default=os.environ.get(
            "EC2_REGION_PRIORITY", "us-east-1,us-west-2,us-east-2"
        ),
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


def offered_in_region(region: str, instance_type: str) -> set[str]:
    """Return the set of AZs that *currently* offer ``instance_type``."""

    ec2 = boto3.client("ec2", region_name=region)
    paginator = ec2.get_paginator("describe_instance_type_offerings")
    azs: set[str] = set()
    for page in paginator.paginate(
        LocationType="availability-zone",
        Filters=[{"Name": "instance-type", "Values": [instance_type]}],
    ):
        for offering in page.get("InstanceTypeOfferings", []):
            azs.add(offering["Location"])
    return azs


def latest_spot_prices(
    region: str,
    instance_type: str,
    lookback_hours: int,
) -> dict[str, dict[str, Any]]:
    """Return the most recent spot price per AZ for ``instance_type``."""

    ec2 = boto3.client("ec2", region_name=region)
    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=lookback_hours)
    paginator = ec2.get_paginator("describe_spot_price_history")
    latest: dict[str, dict[str, Any]] = {}
    for page in paginator.paginate(
        InstanceTypes=[instance_type],
        ProductDescriptions=["Linux/UNIX"],
        StartTime=start,
        EndTime=end,
    ):
        for entry in page.get("SpotPriceHistory", []):
            az = entry["AvailabilityZone"]
            timestamp = entry["Timestamp"]
            if (
                az not in latest
                or timestamp > latest[az]["timestamp"]
            ):
                latest[az] = {
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
    for region in regions:
        for instance_type in instance_types:
            try:
                offered_azs = offered_in_region(region, instance_type)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                print(
                    f"WARN: describe_instance_type_offerings failed in {region} "
                    f"for {instance_type}: {exc}",
                    file=sys.stderr,
                )
                continue
            if not offered_azs:
                continue
            try:
                prices = latest_spot_prices(region, instance_type, lookback_hours)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                print(
                    f"WARN: describe_spot_price_history failed in {region} "
                    f"for {instance_type}: {exc}",
                    file=sys.stderr,
                )
                continue
            for az in offered_azs:
                if az not in prices:
                    continue
                spot_price = prices[az]["price"]
                candidates.append(
                    {
                        "region": region,
                        "instance_type": instance_type,
                        "availability_zone": az,
                        "spot_price_usd_per_hour": spot_price,
                        "max_bid_usd_per_hour": round(
                            spot_price * max_price_multiplier, 4
                        ),
                        "observed_at_utc": prices[az]["timestamp"].isoformat(),
                        "priority_region_index": regions.index(region),
                        "priority_instance_index": instance_types.index(
                            instance_type
                        ),
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
    regions = [r.strip() for r in args.regions.split(",") if r.strip()]
    instance_types = [
        t.strip() for t in args.instance_types.split(",") if t.strip()
    ]
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
