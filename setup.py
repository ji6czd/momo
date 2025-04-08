from setuptools import setup, find_packages
setup(
    name='momoPy',  # パッケージ名（pip listで表示される）
    version="0.0.1",  # バージョン
    description="Japanese Text to Braille Converter Library",  # 説明
    author='KIRIAKE Masanori',  # 作者名
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'protobuf',  # Protobufに依存
        'SudachiPy',  # SudachiPyに依存
        'SudachiDict_core',  # SudachiDict_coreに依存
    ],
    license='Apach'  # ライセンス
)
