import setuptools

setuptools.setup(
    name="stock-prediction package",
    version="0.1.0",
    author="Ivan Manhique",
    author_email="viriatomanhique@gmail.com",
    description="Prediction Apple stock using ARIMA",
    long_description="",
    packages=setuptools.find_packages(),
    python_requires=">=3.10.12",
    dependencies=["pydantic==2.9.2"]
)