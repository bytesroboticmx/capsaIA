#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

# Hide the content between <!-- SETUPTOOLS_LONG_DESCRIPTION_HIDE_BEGIN --> and
# <!-- SETUPTOOLS_LONG_DESCRIPTION_HIDE_END --> tags in the README
while True:
    start_tag = '<!-- SETUPTOOLS_LONG_DESCRIPTION_HIDE_BEGIN -->'
    end_tag = '<!-- SETUPTOOLS_LONG_DESCRIPTION_HIDE_END -->'
    start = readme.find(start_tag)
    end = readme.find(end_tag)
    if start == -1:
        assert end == -1, 'there should be a balanced number of start and ends'
        break
    else:
        assert end != -1, 'there should be a balanced number of start and ends'
        readme = readme[:start] + readme[end + len(end_tag):]

requirements = [ ]
setup_requirements = [ ]
test_requirements = [ ]

setup(
    author="Themis AI",
    author_email='info@themisai.io',
    python_requires='>=3.5',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    description="A data- and model-agnostic neural network wrapper for risk-aware decision making",
    long_description=readme,
    long_description_content_type='text/markdown',
    install_requires=requirements,
    license="GNU Affero General Public License v3.0", #TODO: update
    include_package_data=True,
    keywords='capsa',
    name='capsa',
    packages=find_packages(include=['capsa', 'capsa.*']),
    setup_requires=setup_requirements,
    url='https://github.com/themis-ai/capsa',
    download_url = 'https://github.com/themis-ai/capsa/archive/refs/tags/0.1.1.tar.gz',
    version='0.1.1',
    zip_safe=False,
)
