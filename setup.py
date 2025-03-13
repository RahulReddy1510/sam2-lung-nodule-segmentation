"""
Setup script for sam2-lung-nodule-seg.

This package provides uncertainty-aware lung nodule segmentation via
SAM2 fine-tuning with temporal consistency and Monte Carlo Dropout.
"""

from pathlib import Path

from setuptools import find_packages, setup

# Read requirements from requirements.txt
_req_path = Path(__file__).parent / "requirements.txt"
with open(_req_path, encoding="utf-8") as f:
    requirements = [
        line.strip() for line in f if line.strip() and not line.startswith("#")
    ]

# Read long description from README
_readme_path = Path(__file__).parent / "README.md"
with open(_readme_path, encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="sam2-lung-nodule-seg",
    version="1.0.0",
    author="Rahul Reddy Koulury",
    author_email="koulury2004@gmail.com",
    description=(
        "Uncertainty-aware lung nodule segmentation via SAM2 fine-tuning "
        "with temporal consistency and Monte Carlo Dropout"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rahulkoulury/sam2-lung-nodule-segmentation",
    packages=find_packages(exclude=["tests*", "notebooks*", "scripts*"]),
    python_requires=">=3.10",
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
    ],
    keywords=[
        "lung nodule",
        "segmentation",
        "SAM2",
        "Monte Carlo Dropout",
        "uncertainty quantification",
        "CT imaging",
        "LUNA16",
        "medical imaging",
        "deep learning",
    ],
    entry_points={
        "console_scripts": [
            "luna16-preprocess=data.luna16_preprocessing:main",
            "sam2-train=training.train:main",
            "sam2-evaluate=evaluation.evaluate:main",
        ]
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.ui"],
    },
    zip_safe=False,
    project_urls={
        "Bug Reports": (
            "https://github.com/rahulkoulury/sam2-lung-nodule-segmentation/issues"
        ),
        "Source": ("https://github.com/rahulkoulury/sam2-lung-nodule-segmentation"),
    },
)
