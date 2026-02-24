"""Entry-point script to label radiology reports."""
import sys, os
negbio_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'NegBio'))
if negbio_path not in sys.path:
    sys.path.insert(0, negbio_path)

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

chexpert_labeler_path = os.path.abspath(os.path.dirname(__file__))
if chexpert_labeler_path not in sys.path:
    sys.path.insert(0, chexpert_labeler_path)

import pandas as pd
from args import ArgParser
from loader import Loader
from stages import Extractor, Classifier, Aggregator
from constants import *
import bioc
from NegBio.negbio.pipeline import text2bioc

def label_return_df(args, reports):
    loader = Loader(None, args.sections_to_extract, args.extract_strict)
    loader.reports = [r.strip().replace('\n', ' ') for r in reports]

    collection = bioc.BioCCollection()
    for i, report in enumerate(loader.reports):
        clean_report = loader.clean(report)
        document = text2bioc.text2document(str(i), clean_report)
        if loader.sections_to_extract:
            document = loader.extract_sections(document)
        split_document = loader.splitter.split_doc(document)
        assert len(split_document.passages) == 1, (
            'Each document must be given as a single passage.')
        collection.add_document(split_document)
    loader.collection = collection

    extractor = Extractor(args.mention_phrases_dir,
                          args.unmention_phrases_dir,
                          verbose=args.verbose)
    classifier = Classifier(args.pre_negation_uncertainty_path,
                            args.negation_path,
                            args.post_negation_uncertainty_path,
                            verbose=args.verbose)
    aggregator = Aggregator(CATEGORIES,
                            verbose=args.verbose)

    extractor.extract(loader.collection)
    classifier.classify(loader.collection)
    labels = aggregator.aggregate(loader.collection)

    df = pd.DataFrame({REPORTS: loader.reports})
    for index, category in enumerate(CATEGORIES):
        df[category] = labels[:, index]
    return df



def write(reports, labels, output_path, verbose=False):
    """Write labeled reports to specified path."""
    labeled_reports = pd.DataFrame({REPORTS: reports})
    for index, category in enumerate(CATEGORIES):
        labeled_reports[category] = labels[:, index]

    if verbose:
        print(f"Writing reports and labels to {output_path}.")
    labeled_reports[[REPORTS] + CATEGORIES].to_csv(output_path,
                                                   index=False)


def label(args):
    """Label the provided report(s)."""

    loader = Loader(args.reports_path,
                    args.sections_to_extract,
                    args.extract_strict)

    extractor = Extractor(args.mention_phrases_dir,
                          args.unmention_phrases_dir,
                          verbose=args.verbose)
    classifier = Classifier(args.pre_negation_uncertainty_path,
                            args.negation_path,
                            args.post_negation_uncertainty_path,
                            verbose=args.verbose)
    aggregator = Aggregator(CATEGORIES,
                            verbose=args.verbose)

    # Load reports in place.
    loader.load()
    # Extract observation mentions in place.
    extractor.extract(loader.collection)
    # Classify mentions in place.
    classifier.classify(loader.collection)
    # Aggregate mentions to obtain one set of labels for each report.
    labels = aggregator.aggregate(loader.collection)

    write(loader.reports, labels, args.output_path, args.verbose)


if __name__ == "__main__":
    parser = ArgParser()
    label(parser.parse_args())
