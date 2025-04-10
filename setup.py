from setuptools import setup, find_packages

setup(
    name="robust_tracker",
    version="0.1.0",
    description="A robust face tracking model using Kalman filtering and IoU",
    author="Rohit Kumar Nayak",
    author_email="rohit.nayak@oditeksolutions.com",
    url="https://github.com/Rohit252001/robust_tracker",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "filterpy>=1.4.5",
        "scipy>=1.7.0"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
