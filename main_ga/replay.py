import json
import os
import csv


from apollo.ApolloContainer import ApolloContainer
from config import APOLLO_ROOT, HD_MAP_PATH
from framework.model.ModelManager import NPCModel
from framework.scenario import Scenario, NPCSection
from framework.scenario.ScenarioRunner import ScenarioRunner
from framework.scenario.ad_agents import ADSection, ADAgent
from framework.scenario.tc_config import TCSection
from hdmap.MapParser import MapParser

p = ''
def get_base(path:json):
    parts = path.split('/')
    mes = parts[8:10]
    jsonPath = p + '/'.join(mes) + '/c.json'
    return jsonPath

def get_length():
    json_set = set()
    with open(p + '/summary.csv', mode='r', newline='') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            if row['main_type'] == 'collision':
                path = row['record_path']
                json_path = get_base(path)
                json_set.add(json_path)
    # print(json_set)
    print(len(list(json_set)))

def replay_list():
    json_set = set()
    with open(p+'summary.csv', mode='r', newline='') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            if row['main_type'] == 'collision':
                path = row['record_path']
                json_path = get_base(path)
                json_set.add(json_path)

    container = ApolloContainer(APOLLO_ROOT, f'ROUTE_0')
    container.start_instance()
    container.start_dreamview()
    print(f'Dreamview at http://{container.ip}:{container.port}')
    srunner = ScenarioRunner(container)

    json_list = list(sorted(json_set))
    # print(json_list)
    for i in range(len(json_list)):
        print(json_list[i])
        x = Scenario.get_from_json_replay(json_list[i])
        srunner.set_scenario(x)
        srunner.init_scenario()
        srunner.replay_scenario()

def replay_one():
    container = ApolloContainer(APOLLO_ROOT, f'ROUTE_0')
    container.start_instance()
    container.start_dreamview()
    print(f'Dreamview at http://{container.ip}:{container.port}')
    srunner = ScenarioRunner(container)
    json_path = ''
    x = Scenario.get_from_json_replay(json_path)

    srunner.set_scenario(x)
    srunner.init_scenario()
    srunner.replay_scenario()



if __name__ == '__main__':
    # replay_list()
    # replay_one()
    get_length()