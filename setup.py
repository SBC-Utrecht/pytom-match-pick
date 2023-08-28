import setuptools

# readme fetch
with open('README.md', 'r') as f:
    long_description = f.read()

setuptools.setup(
    name='pytom-template-matching-gpu',
    packages=['pytom_tm', 'pytom_tm.angle_lists'],
    package_dir={'': 'src'},
    version='0.1',
    description='GPU template matching from PyTOM as a lightweight pip package',
    long_description=long_description,
    long_description_content_type='text/markdown',
    license='GPLv3',
    author='McHaillet (Marten Chaillet)',
    url='https://github.com/McHaillet/pytom-template-matching-gpu',
    platforms=['any'],
    python_requires='>=3.9',
    install_requires=[
        'numpy',
        'cupy',
        'voltools',
        'tqdm',
        'mrcfile',
        'starfile',
        'importlib_resources'
    ],
    package_data={
        'pytom_tm.angle_lists': ['*.txt'],
    },
    test_suite='tests',
    scripts=[
        'src/bin/pytom_create_mask.py',
        'src/bin/pytom_create_template.py',
        'src/bin/pytom_match_template.py',
        'src/bin/pytom_extract_candidates.py',
        'src/bin/pytom_estimate_roc.py',
        'src/bin/pytom_merge_stars.py',
    ],
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: GPLv3 License',
        'Programming Language :: Python :: 3 :: Only',
    ]
)
