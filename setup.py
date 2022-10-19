import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="transcript transformer",
    version="0.0.2",
    author="Jim Clauwaert",
    author_email="jim.clauwaert@ugent.be",
    packages=["transcript_transformer"],
    description="A package combining sequence and ribosome profiling data for performing deep learning on the transcriptome",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jdcla/transcript_transformer",
    license='MIT',
    python_requires='>=3.9',
    install_requires=[
         "pytorch-lightning>=1.7.0", "torchmetrics>=0.9.3", "performer-pytorch>=1.1.4", "h5py>=3.6.0", "numpy>=1.21.0", "scipy>=1.7.3", "h5max>=0.1.1"
    ],
    entry_points = {
        'console_scripts': ['transcript_transformer=transcript_transformer.transcript_transformer:main']
    }
)
