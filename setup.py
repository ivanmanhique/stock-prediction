import setuptools

setuptools.setup(
    name="stock-prediction-package",
    version="0.1.0",
    author="Ivan Manhique",
    author_email="vriationmanhique@gmail.com",
    description="Prediction of Apple stock using ARIMA",
    long_description="This package provides tools to predict Apple stock prices using ARIMA and other methods.",
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    python_requires=">=3.10.12",
    install_requires=[
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "statsmodels>=0.12.0",
        "prophet>=1.0",
        "scipy>=1.7.0",
        "dvc>=3.0.5",
    ],
)
