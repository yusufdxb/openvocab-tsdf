from setuptools import find_packages, setup

package_name = "openvocab_tsdf_node"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", ["launch/bringup.launch.py"]),
        (f"share/{package_name}/config", ["config/default.yaml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Yusuf",
    maintainer_email="yusuf@mewtwo",
    description="ROS 2 node wrapping the openvocab-tsdf pipeline.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "grounding_node = openvocab_tsdf_node.grounding_node:main",
        ],
    },
)
