from quick_pypi.deploy import *

auto_deploy(
    cwd=os.path.dirname(os.path.realpath(__file__)),
    name="tm-eval",
    long_name="Topic Modeling Evaluation",
    description="Topic Modeling Evaluation",
    long_description="A toolkit to quickly evaluate topic model goodness over number of topics",
    src_root="src",
    dists_root=f"dists",
    pypi_token='../pypi_upload_token.txt',
    test=False,
    version="0.0.1",
    project_url="http://github.com/dhchenx/tm-eval",
    author_name="Donghua Chen",
    author_email="douglaschan@126.com",
    requires="gensim", # use ; for multiple requires
    license='MIT',
    license_filename='LICENSE',
    keywords="topic modeling, metrics",
    github_username="dhchenx",
    readme_path="README.md",
   # console_scripts=""
)

