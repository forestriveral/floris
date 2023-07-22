import yaml, json
from floris.utilities import load_yaml
from dataclasses import dataclass, field, fields
import attrs
from attr import attrs, attrib, fields, Factory
from attrs import define, field
import typing


json_file = './Vesta_2MW.json'
yaml_file = './Vesta_2MW.yaml'


def load_json(path):
    with open(path,'r+') as load_f:
            config = json.load(load_f)
    return config


def save_yaml(data, save_path="test.yaml"):
    with open(save_path, "w") as f:
        yaml.dump(data, f, sort_keys=False, indent=2)


@dataclass
class Lang:
    """a dataclass that describes a programming language"""
    name: str = 'python'
    strong_type: bool = True
    static_type: bool = False
    age: int = 28


if __name__ == "__main__":
    # jconfig = load_json(json_file)
    # power_table = jconfig["turbine"]["properties"]["power_thrust_table"]
    # print(power_table)

    # yconfig = load_yaml(yaml_file)
    # print(yconfig["power_thrust_table"])
    # yconfig["power_thrust_table"] = power_table
    # save_yaml(yconfig, "./test.yaml")

    from typing import NewType

    UserId = NewType('UserId', int)


    def name_by_id(user_id: UserId) -> str:
        print(user_id)

    test_num = UserId('user')  # Fails type check
    print(type(test_num))
    num = UserId(5)  # type: int
    print(num)

    name_by_id(42)  # Fails type check
    name_by_id(UserId(42))  # OK

    print(type(UserId(5)))
