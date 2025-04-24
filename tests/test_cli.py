import os
import pytest
from itertools import product
import subprocess
import h5py
from pathlib import Path


@pytest.fixture(autouse=True)
def setup_module():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))


def remove_if_exists(file_paths):
    """Remove the file if it exists."""
    for file_path in file_paths:
        if Path(file_path).exists():
            os.remove(file_path)


def assert_if_exists(file_paths):
    """Assert that the file exists."""
    for file_path in file_paths:
        assert Path(file_path).exists(), f"{file_path} was not created"


def test_cli_version():
    result = subprocess.run(["ribotie", "--version"], capture_output=True, text=True)

    # Check that the return code is zero (indicating success)
    assert result.returncode == 0

    result = subprocess.run(
        ["tis_transformer", "--version"], capture_output=True, text=True
    )

    # Check that the return code is zero (indicating success)
    assert result.returncode == 0


def test_cli_help():
    result = subprocess.run(["ribotie", "--help"], capture_output=True, text=True)

    assert result.returncode == 0
    assert "ribotie" in result.stdout
    # Add more checks based on expected help output


# TIS Transformer
def test_sequence_data_loading():
    remove_if_exists(["dbs/tis_transformer_loading.h5"])

    command = [
        "tis_transformer",
        "configs/tis_transformer_loading.yml",
    ]

    result = subprocess.run(command, check=True, text=True, capture_output=True)
    file_path = Path("dbs/tis_transformer_loading.h5")
    assert file_path.exists(), "HDF5 file was not created"

    f = h5py.File(file_path, "r")
    assert "transcript" in f.keys(), "HDF5 file does not contain 'transcript' group"
    assert len(f["transcript"].keys()) > 3, "HDF5 file does not contain expected keys"
    assert len(f["transcript/seqname"]) > 0, "HDF5 file does not contain 'seqname' data"


def test_sequence_training():
    files = [
        "out/tis_transformer_seq.npy",
        "out/tis_transformer_seq.csv",
        "out/tis_transformer_seq.gtf",
        "out/tis_transformer_seq.unfiltered.csv",
        "out/tis_transformer_seq.unfiltered.gtf",
    ]
    remove_if_exists(files)

    command = [
        "tis_transformer",
        "configs/tis_transformer_training.yml",
    ]

    subprocess.run(command, check=True, text=True, capture_output=True)

    assert_if_exists(files)


# RiboTIE
def test_ribotie_sam_data_loading():
    remove_if_exists(["dbs/ribotie_sam_loading.h5"])

    command = [
        "ribotie",
        "configs/ribotie_sam_loading.yml",
    ]

    result = subprocess.run(command, check=True, text=True, capture_output=True)
    file_path = Path("dbs/ribotie_sam_loading.h5")
    assert file_path.exists(), "HDF5 file was not created"
    f = h5py.File(file_path, "r")["transcript"]
    assert "riboseq" in f.keys(), "HDF5 file does not contain 'ribo' group"
    assert (
        len(f["riboseq/sample_1/5"].keys()) > 3
    ), "HDF5 file does not contain expected keys"
    assert (
        len(f["riboseq/sample_1/5/data"]) > 0
    ), "HDF5 file does not contain 'data' entries"


def test_ribotie_bam_data_loading():
    remove_if_exists(["dbs/ribotie_bam_loading.h5"])

    command = [
        "ribotie",
        "configs/ribotie_bam_loading.yml",
    ]

    subprocess.run(command, check=True, text=True, capture_output=True)
    file_path = Path("dbs/ribotie_bam_loading.h5")
    assert file_path.exists(), "HDF5 file was not created"
    f = h5py.File(file_path, "r")["transcript"]
    assert "riboseq" in f.keys(), "HDF5 file does not contain 'ribo' group"
    assert (
        len(f["riboseq/sample_1/5"].keys()) > 3
    ), "HDF5 file does not contain expected keys"
    assert (
        len(f["riboseq/sample_1/5/data"]) > 0
    ), "HDF5 file does not contain 'data' entries"


def test_ribotie_pretraining():
    files = [
        f"out/pretrain_f{i}.{ext}"
        for i, ext in list(product(range(2), ["npy", "csv", "gtf"]))
    ] + [
        f"out/pretrain_f{i}.unfiltered.{ext}"
        for i, ext in list(product(range(2), ["csv", "gtf"]))
    ]

    remove_if_exists(files)

    command = [
        "ribotie",
        "configs/ribotie_pretraining.yml",
    ]
    subprocess.run(command, check=True, text=True, capture_output=True)

    assert_if_exists(files)


def test_ribotie_training():
    files = [
        f"out/sample_{i}.{ext}"
        for i, ext in list(product(range(1, 4), ["npy", "csv", "gtf"]))
    ] + [
        f"out/sample_{i}.unfiltered.{ext}"
        for i, ext in list(product(range(1, 4), ["csv", "gtf"]))
    ]
    remove_if_exists(files)

    command = [
        "ribotie",
        "configs/ribotie_training.yml",
    ]
    subprocess.run(command, check=True, text=True, capture_output=True)

    assert_if_exists(files)
