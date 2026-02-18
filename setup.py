from setuptools import setup, find_packages

setup(
    name='trustcall',
    version='1.0.0',
    description='Deepfake voice detection via neural vocoder artifact analysis',
    author='Akshat Agrawal',
    python_requires='>=3.8',
    packages=find_packages(exclude=['venv', 'outputs', 'data']),
    install_requires=[
        'torch>=1.12.0',
        'torchaudio>=0.12.0',
        'librosa>=0.9.2',
        'numpy>=1.21.0',
        'scikit-learn>=1.0.0',
        'matplotlib>=3.5.0',
        'pyyaml>=6.0',
        'tqdm>=4.64.0',
        'streamlit>=1.20.0',
        'soundfile>=0.11.0',
    ],
    entry_points={
        'console_scripts': [
            'trustcall-train=main:main',
            'trustcall-eval=eval:main',
            'trustcall-bench=benchmark:main',
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Multimedia :: Sound/Audio :: Analysis',
    ],
)
