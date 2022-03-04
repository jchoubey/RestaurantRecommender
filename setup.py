import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="restaurantrecommender",
    version="0.0.1",
    author="Poornima Muthukumar",
    author_email="muthupoo2@uw.edu",
    description="Package for Recommending Restaurant",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Poornima-Muthukumar/RestaurantRecommender",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=['restaurantrecommender'],
    python_requires=">=3.9",
    install_requires=['numpy', 'pandas', 'matplotlib', 'scipy', 'implicit', 'sklearn']
)
