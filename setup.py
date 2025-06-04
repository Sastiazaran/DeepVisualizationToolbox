from setuptools import setup, find_packages

setup(
    name="tf-feature-vis",
    version="0.1.0",
    description="TensorFlow Feature Visualization Toolbox",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "tensorflow>=2.4.0",
        "numpy>=1.19.0",
        "scipy>=1.6.0",
        "matplotlib>=3.3.0",
        "pillow>=8.0.0",
        "opencv-python>=4.5.0",
        "scikit-image>=0.18.0",
        "PyQt5>=5.15.0",
    ],
    entry_points={
        "console_scripts": [
            "tf-feature-vis=tf_vis.app:main",
        ],
    },
)
