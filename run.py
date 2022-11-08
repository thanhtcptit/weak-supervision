# -*- coding: utf-8 -*-
import os
import sys

import pkgutil
import importlib


def import_submodules(package_name: str) -> None:
    importlib.invalidate_caches()
    sys.path.append(".")

    module = importlib.import_module(package_name)
    path = getattr(module, "__path__", [])
    path_string = "" if not path else path[0]

    for module_finder, name, _ in pkgutil.walk_packages(path):
        if path_string and module_finder.path != path_string:
            continue
        subpackage = f"{package_name}.{name}"
        import_submodules(subpackage)


import_submodules("src")
from src.utils.params import Params
from src.utils.args_parser import Subcommand, ArgumentParserWithDefaults


def main(subcommand_overrides={}):
    parser = ArgumentParserWithDefaults()

    subparsers = parser.add_subparsers(title="Commands", metavar="")

    subcommands = {
        "train": Train(),
        "eval": Evaluate(),
        "bench": Benchmark(),
        "hyp": HyperparamsSearch(),
        "export": ExportModel(),
        **subcommand_overrides
    }

    for name, subcommand in subcommands.items():
        subparser = subcommand.add_subparser(name, subparsers)
        if name != "configure":
            subparser.add_argument("--include-package",
                                   type=str,
                                   action="append",
                                   default=[],
                                   help="additional packages to include")

    args = parser.parse_args()
    if "func" in dir(args):
        for package_name in getattr(args, "include_package", ()):
            import_submodules(package_name)
        args.func(args)
    else:
        parser.print_help()


class Train(Subcommand):
    def add_subparser(self, name, parser):
        description = """Train the specified model on the specified dataset"""
        subparser = parser.add_parser(name, description=description,
                                      help=description)

        subparser.add_argument(
            "config_path", type=str,
            help="path to parameter file describing the model to be trained")
        subparser.add_argument(
            "-s", "--save_dir", type=str, default="",
            help="directory in which to save the model and its logs")
        subparser.add_argument(
            "-r", "--recover", action="store_true",
            help="recover training from the state in serialization_dir")
        subparser.add_argument(
            "-f", "--force", action="store_true",
            help="force override serialization dir")
        subparser.add_argument(
            "-v", "--verbose", action="store_true",
            help="show training log")

        subparser.set_defaults(func=train_model)

        return subparser


def train_model(args):
    from src.train import train as func
    return func(args.config_path, args.save_dir, args.recover, args.force,
                args.verbose)


class Evaluate(Subcommand):
    def add_subparser(self, name, subparsers):
        description = "Run evaluation"
        subparser = subparsers.add_parser(name, description=description,
                                          help=description)

        subparser.add_argument(
            "config_path", type=str,
            help=("path to the config file"))

        subparser.set_defaults(func=eval_model)
        return subparser


def eval_model(args):
    from src.train import eval as func
    return func(args.config_path)


class Benchmark(Subcommand):
    def add_subparser(self, name, parser):
        description = """Benchmark the specified model on the specified dataset"""
        subparser = parser.add_parser(name, description=description,
                                      help=description)

        subparser.add_argument(
            "config_path", type=str,
            help="path to parameter file describing the model to be trained")
        subparser.add_argument(
            "-s", "--save_dir", type=str, default="train_logs/bench",
            help="directory in which to save the model and its logs")

        subparser.set_defaults(func=benchmark_model)

        return subparser


def benchmark_model(args):
    from src.benchmark import main as func
    return func(args.config_path, args.save_dir)


class HyperparamsSearch(Subcommand):
    def add_subparser(self, name, subparsers):
        description = "Run hyperparams search"
        subparser = subparsers.add_parser(name, description=description,
                                          help=description)

        subparser.add_argument(
            "config_path", type=str,
            help="path to the json config file")
        subparser.add_argument(
            "-n", "--num_trials", type=str, default=None,
            help="number of trials to run")
        subparser.add_argument(
            "-f", "--force", action="store_true",
            help="force override serialization dir")

        subparser.set_defaults(func=hyperparams_search)
        return subparser


def hyperparams_search(args):
    raise NotImplementedError()


class ExportModel(Subcommand):
    def add_subparser(self, name, parser):
        description = """Export model for serving"""
        subparser = parser.add_parser(name, description=description,
                                      help=description)

        subparser.add_argument(
            "checkpoint_path", type=str, help="path to model checkpoint")
        subparser.add_argument(
            "-o", "--output_dir", type=str, help="path to output dir")
        subparser.set_defaults(func=export_model)

        return subparser


def export_model(args):
    raise NotImplementedError()


def run():
    main()


if __name__ == "__main__":
    run()
