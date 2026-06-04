from collections import Counter

from flwr_datasets.partitioner import IidPartitioner, PathologicalPartitioner
from blackcatt import wm_config

try:
    from flwr_datasets.partitioner import ShardPartitioner
except ImportError:
    ShardPartitioner = None


def label_column(dataset):
    if dataset == "CIFAR100":
        return "fine_label"
    return "label"


def make_partitioner(num_partitions, dataset):
    label_col = label_column(dataset)
    mode = getattr(wm_config, "partition_mode", "iid")

    if mode == "iid":
        return IidPartitioner(num_partitions=num_partitions)

    if mode == "pathological":
        kwargs = dict(
            num_partitions=num_partitions,
            partition_by=label_col,
            num_classes_per_partition=wm_config.num_classes_per_client,
            shuffle=True,
            seed=wm_config.partition_seed,
        )

        try:
            return PathologicalPartitioner(
                **kwargs,
                class_assignment_mode=wm_config.class_assignment_mode,
            )
        except (TypeError, ValueError):
            return PathologicalPartitioner(**kwargs)

    if mode == "shard":
        if ShardPartitioner is None:
            raise ImportError("ShardPartitioner is unavailable. Use pathological mode.")

        return ShardPartitioner(
            num_partitions=num_partitions,
            partition_by=label_col,
            num_shards_per_partition=wm_config.num_shards_per_client,
            shuffle=True,
            seed=wm_config.partition_seed,
        )

    raise ValueError(f"Unknown partition_mode: {mode}")


def print_label_distribution(partition, partition_id, dataset):
    if not getattr(wm_config, "print_partition_stats", False):
        return

    label_col = label_column(dataset)
    labels = partition[label_col]
    counter = Counter(labels)

    print(
        f"[Partition {partition_id}] "
        f"size={len(partition)}, "
        f"num_labels={len(counter)}, "
        f"labels={dict(sorted(counter.items()))}"
    )
