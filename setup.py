try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'description': 'Gene predicting software.',
    'author': 'Paul Crook, Saman Azizi',
    'url': 'URL to get it at.',
    'download_url': 'Where to download it.',
    'author_email': 'paul.crook@uranus.uni-freiburg.de',
    'version': '0.1a',
    'install_requires': ['matplotlib', 'numpy', ''],
    'name': 'PrognoGene'
}

setup(**config)
