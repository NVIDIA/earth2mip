from earth2mip import networks, schema
import argparse


def add_model_args(parser: argparse.ArgumentParser, required=False):
    if required:
        parser.add_argument("model", type=str)
    else:
        parser.add_argument("--model", type=str)
    parser.add_argument(
        "--model-metadata",
        type=str,
        help="metadata.json file. Defaults to the metadata.json in the model package.",
        default="",
    )


def model_from_args(args, device):
    if args.model_metadata:
        with open(args.model_metadata) as f:
            metadata = schema.Model.parse_raw(f.read())
    else:
        metadata = None

    return networks.get_model(args.model, device=device, metadata=metadata)
