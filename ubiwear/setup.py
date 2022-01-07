from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='ubiwear',
    version='0.0.1',
    url='https://github.com/stergiosbamp/deep-physical-activity-prediction/ubiwear/',
    description='A Python library for pre-processing ubiquitous aggregated self-tracking data',
    long_description=long_description,
    long_description_content_type='text/markdown',
    license_files=['LICENSE'],
    author='Stergios Bampakis',
    author_email='bampakis.stergios@gmail.com',
    python_requires='>=3.8',
    classifiers=[
        'Programming Language :: Python :: 3.8',
        'Development Status :: 3 - Alpha',
        'Operating System :: OS Independent'
    ],
    install_requires=['pandas', 'scikit-learn'],
    packages=['ubiwear']
)
