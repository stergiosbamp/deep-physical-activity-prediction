from setuptools import setup

setup(
    name='ubiwear-stergiosbamp',
    version='',
    url='https://github.com/stergiosbamp/deep-physical-activity-prediction',
    license='',
    author='Stergios Bampakis',
    author_email='bampakis.stergios@gmail.com',
    python_requires='>=3.8',
    classifiers=[
        'Programming Language :: Python :: 3.8',
        'Development Status :: 3 - Alpha',
        'Operating System :: OS Independent'
    ],
    install_requires=['pandas', 'scikit-learn'],
    packages=['ubiwear'],
    description='A Python library for pre-processing ubiquitous aggregated self-tracking data'
)
