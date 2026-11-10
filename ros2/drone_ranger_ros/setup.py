from glob import glob

from setuptools import find_packages, setup

package_name = "drone_ranger_ros"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Luke Andrew",
    description="Drone Ranger as a ROS 2 node.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "drone_ranger_node = drone_ranger_ros.node:main",
            "make_bag = drone_ranger_ros.make_bag:main",
        ],
    },
)
