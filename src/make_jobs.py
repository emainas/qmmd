#!/usr/bin/env python3

from dataclasses import dataclass

@dataclass
class SlurmConfiguration:
    """Class for spefifying the configuration of a DCDFTBMD run with slurm."""
    number_of_nodes:
    number_of_tasks:
    cpus_per_task:
    partition:
    name:
    time:



def main():
    pass

if __name__ == "__main__":
    main()
