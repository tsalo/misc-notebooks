"""Reorganize Neurosynth coordinates files."""
import os.path as op

import pandas as pd
from scipy import sparse


def reorganize_coordinates_file(in_dir, out_dir, in_file, dict_):
    print(f"Processing {in_file}")
    in_file = op.join(in_dir, in_file)
    coordinates_file = op.join(out_dir, dict_["coordinates"])
    metadata_file = op.join(out_dir, dict_["metadata"])

    if op.isfile(metadata_file):
        print(f"\tSkipping {metadata_file}")
        return

    df = pd.read_table(in_file)
    df = df.sort_values(by="id")
    coordinates_df = df[["id", "table_id", "table_num", "peak_id", "x", "y", "z"]]
    metadata_df = df[["id", "doi", "space", "title", "authors", "year", "journal"]]
    metadata_df = metadata_df.drop_duplicates(subset="id", keep="first", ignore_index=True)
    coordinates_df.to_csv(coordinates_file, sep="\t", line_terminator="\n", index=False)
    metadata_df.to_csv(metadata_file, sep="\t", line_terminator="\n", index=False)


def reorganize_feature_file(in_dir, out_dir, in_file, dict_):
    print(f"Processing {in_file}")
    in_file = op.join(in_dir, in_file)
    vocab_file = op.join(out_dir, dict_["vocab"])
    metadata_file = op.join(out_dir, dict_["metadata"])
    features_file = op.join(out_dir, dict_["features"])

    if op.isfile(features_file):
        print("\tSkipping")
        return

    assert not op.isfile(vocab_file)

    try:
        original_df = pd.read_table(in_file, index_col="pmid")
    except ValueError:
        original_df = pd.read_table(in_file, index_col="id")

    original_df.index = original_df.index.astype(str)
    feature_ids = original_df.index.tolist()

    metadata_df = pd.read_table(metadata_file)
    metadata_ids = metadata_df["id"].astype(str).tolist()

    in_metadata_but_not_feature = list(set(metadata_ids) - set(feature_ids))
    in_feature_but_not_metadata = list(set(feature_ids) - set(metadata_ids))

    if in_metadata_but_not_feature:
        raise Exception(
            f"{len(in_metadata_but_not_feature)} found in IDs file but not {in_file}"
        )

    if in_feature_but_not_metadata:
        raise Exception(
            f"{len(in_feature_but_not_metadata)} found in {in_file} but not IDs file"
        )

    # Ensure same order
    sorted_df = original_df.loc[metadata_ids]

    # Now split into data, vocab, and ids
    feature_data = sorted_df.to_numpy()
    feature_vocab = sorted_df.columns.tolist()

    # Output vocab
    with open(vocab_file, "w") as fo:
        fo.write("\n".join(feature_vocab))

    # Convert to Compressed Sparse Column format sparse matrix and save to file
    feature_data_sparse = sparse.csc_matrix(feature_data)
    sparse.save_npz(features_file, feature_data_sparse, compressed=True)


if __name__ == "__main__":
    DATABASE_FILES = {
        "current_data/database.txt": {
            "coordinates": "data-neurosynth_version-7_coordinates.tsv.gz",
            "metadata": "data-neurosynth_version-7_metadata.tsv.gz",
        },
        "archive/data_0.6.July_2015/database.txt": {
            "coordinates": "data-neurosynth_version-6_coordinates.tsv.gz",
            "metadata": "data-neurosynth_version-6_metadata.tsv.gz",
        },
        "archive/data_0.5.February_2015/database.txt": {
            "coordinates": "data-neurosynth_version-5_coordinates.tsv.gz",
            "metadata": "data-neurosynth_version-5_metadata.tsv.gz",
        },
        "archive/data_0.4.September_2014/database.txt": {
            "coordinates": "data-neurosynth_version-4_coordinates.tsv.gz",
            "metadata": "data-neurosynth_version-4_metadata.tsv.gz",
        },
        "archive/data_0.3.April_2014/database.txt": {
            "coordinates": "data-neurosynth_version-3_coordinates.tsv.gz",
            "metadata": "data-neurosynth_version-3_metadata.tsv.gz",
        },
        # "archive/data_0.2.May_2013/database.txt": {
        #     "coordinates": "data-neurosynth_version-2_coordinates.tsv.gz",
        #     "metadata": "data-neurosynth_version-2_metadata.tsv.gz",
        # },
    }
    FEATURE_FILES = {
        "current_data/features.txt": {
            "coordinates": "data-neurosynth_version-7_coordinates.tsv.gz",
            "vocab": "data-neurosynth_version-7_vocab-terms_vocabulary.txt",
            "metadata": "data-neurosynth_version-7_metadata.tsv.gz",
            "features": (
                "data-neurosynth_version-7_vocab-terms_source-abstract_type-tfidf_features.npz"
            ),
        },
        "archive/data_0.6.July_2015/features.txt": {
            "coordinates": "data-neurosynth_version-6_coordinates.tsv.gz",
            "vocab": "data-neurosynth_version-6_vocab-terms_vocabulary.txt",
            "metadata": "data-neurosynth_version-6_metadata.tsv.gz",
            "features": (
                "data-neurosynth_version-6_vocab-terms_source-abstract_type-tfidf_features.npz"
            ),
        },
        "archive/data_0.5.February_2015/features.txt": {
            "coordinates": "data-neurosynth_version-5_coordinates.tsv.gz",
            "vocab": "data-neurosynth_version-5_vocab-terms_vocabulary.txt",
            "metadata": "data-neurosynth_version-5_metadata.tsv.gz",
            "features": (
                "data-neurosynth_version-5_vocab-terms_source-abstract_type-tfidf_features.npz"
            ),
        },
        "archive/data_0.4.September_2014/features.txt": {
            "coordinates": "data-neurosynth_version-4_coordinates.tsv.gz",
            "vocab": "data-neurosynth_version-4_vocab-terms_vocabulary.txt",
            "metadata": "data-neurosynth_version-4_metadata.tsv.gz",
            "features": (
                "data-neurosynth_version-4_vocab-terms_source-abstract_type-tfidf_features.npz"
            ),
        },
        "archive/data_0.3.April_2014/features.txt": {
            "coordinates": "data-neurosynth_version-3_coordinates.tsv.gz",
            "vocab": "data-neurosynth_version-3_vocab-terms_vocabulary.txt",
            "metadata": "data-neurosynth_version-3_metadata.tsv.gz",
            "features": (
                "data-neurosynth_version-3_vocab-terms_source-abstract_type-tfidf_features.npz"
            ),
        },
        # "archive/data_0.2.May_2013/features.txt": {
        #     "coordinates": "data-neurosynth_version-2_coordinates.tsv.gz",
        #     "vocab": "data-neurosynth_version-2_vocab-terms_vocabulary.tsv",
        #     "metadata": "data-neurosynth_version-2_ids.tsv",
        #     "features": (
        #         "data-neurosynth_version-2_vocab-terms_source-abstract_type-tfidf_features.npz"
        #     ),
        # },
    }
    TOPIC_FEATURE_FILES = {
        "topics/v5-topics/analyses/v5-topics-50.txt": {
            "coordinates": "data-neurosynth_version-7_coordinates.tsv.gz",
            "vocab": "data-neurosynth_version-7_vocab-LDA50_vocabulary.txt",
            "metadata": "data-neurosynth_version-7_metadata.tsv.gz",
            "features": (
                "data-neurosynth_version-7_vocab-LDA50_source-abstract_type-weight_features.npz"
            ),
        },
        "topics/v5-topics/analyses/v5-topics-100.txt": {
            "coordinates": "data-neurosynth_version-7_coordinates.tsv.gz",
            "vocab": "data-neurosynth_version-7_vocab-LDA100_vocabulary.txt",
            "metadata": "data-neurosynth_version-7_metadata.tsv.gz",
            "features": (
                "data-neurosynth_version-7_vocab-LDA100_source-abstract_type-weight_features.npz"
            ),
        },
        "topics/v5-topics/analyses/v5-topics-200.txt": {
            "coordinates": "data-neurosynth_version-7_coordinates.tsv.gz",
            "vocab": "data-neurosynth_version-7_vocab-LDA200_vocabulary.txt",
            "metadata": "data-neurosynth_version-7_metadata.tsv.gz",
            "features": (
                "data-neurosynth_version-7_vocab-LDA200_source-abstract_type-weight_features.npz"
            ),
        },
        "topics/v5-topics/analyses/v5-topics-400.txt": {
            "coordinates": "data-neurosynth_version-7_coordinates.tsv.gz",
            "vocab": "data-neurosynth_version-7_vocab-LDA400_vocabulary.txt",
            "metadata": "data-neurosynth_version-7_metadata.tsv.gz",
            "features": (
                "data-neurosynth_version-7_vocab-LDA400_source-abstract_type-weight_features.npz"
            ),
        },
        "topics/v4-topics/analyses/v4-topics-50.txt": {
            "coordinates": "data-neurosynth_version-6_coordinates.tsv.gz",
            "vocab": "data-neurosynth_version-6_vocab-LDA50_vocabulary.txt",
            "metadata": "data-neurosynth_version-6_metadata.tsv.gz",
            "features": (
                "data-neurosynth_version-6_vocab-LDA50_source-abstract_type-weight_features.npz"
            ),
        },
        "topics/v4-topics/analyses/v4-topics-100.txt": {
            "coordinates": "data-neurosynth_version-6_coordinates.tsv.gz",
            "vocab": "data-neurosynth_version-6_vocab-LDA100_vocabulary.txt",
            "metadata": "data-neurosynth_version-6_metadata.tsv.gz",
            "features": (
                "data-neurosynth_version-6_vocab-LDA100_source-abstract_type-weight_features.npz"
            ),
        },
        "topics/v4-topics/analyses/v4-topics-200.txt": {
            "coordinates": "data-neurosynth_version-6_coordinates.tsv.gz",
            "vocab": "data-neurosynth_version-6_vocab-LDA200_vocabulary.txt",
            "metadata": "data-neurosynth_version-6_metadata.tsv.gz",
            "features": (
                "data-neurosynth_version-6_vocab-LDA200_source-abstract_type-weight_features.npz"
            ),
        },
        "topics/v4-topics/analyses/v4-topics-400.txt": {
            "coordinates": "data-neurosynth_version-6_coordinates.tsv.gz",
            "vocab": "data-neurosynth_version-6_vocab-LDA400_vocabulary.txt",
            "metadata": "data-neurosynth_version-6_metadata.tsv.gz",
            "features": (
                "data-neurosynth_version-6_vocab-LDA400_source-abstract_type-weight_features.npz"
            ),
        },
    }

    IN_DIR = "/Users/taylor/Downloads/neurosynth-data-e8f27c4a9a44dbfbc0750366166ad2ba34ac72d6/"
    OUT_DIR = "/Users/taylor/Documents/tsalo/neurosynth-data/"

    for k, v in DATABASE_FILES.items():
        reorganize_coordinates_file(IN_DIR, OUT_DIR, k, v)

    for k, v in FEATURE_FILES.items():
        reorganize_feature_file(IN_DIR, OUT_DIR, k, v)

    for k, v in TOPIC_FEATURE_FILES.items():
        reorganize_feature_file(IN_DIR, OUT_DIR, k, v)
