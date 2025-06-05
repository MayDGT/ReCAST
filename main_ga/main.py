import json
import os
import pickle
import time
from copy import deepcopy
from datetime import datetime
from random import random, randint, shuffle

import numpy as np

from apollo.ApolloContainer import ApolloContainer
from apollo.utils import PositionEstimate
from config import HD_MAP_PATH, MAX_ADC_COUNT, APOLLO_ROOT, RECORDS_DIR
from framework.model.ModelManager import NPCModel
from framework.oracles import RecordAnalyzer
from framework.oracles.ViolationTracker import ViolationTracker
from framework.scenario import Scenario, NPCSection
from framework.scenario.ScenarioRunner import ScenarioRunner
from framework.scenario.npc_agents import NPCAgent
from hdmap.MapParser import MapParser
from deap import base, tools, algorithms

adc_start = PositionEstimate('lane_31', 0)
def eval_scenario(ind: Scenario):
    g_name = f'Generation_{ind.gid:05}'
    s_name = f'Scenario_{ind.cid:05}'
    srunner = ScenarioRunner.get_instance()
    srunner.set_scenario(ind)
    srunner.init_scenario()

    runner, adc, npcs = srunner.run_scenario(g_name, s_name, True)

    episodeReward = list()
    for npc in npcs:
        episodeReward.append(npc.reward.get_episode_reward())
    npcRobustness = sum(episodeReward)

    c_name = runner.container.container_name
    r_name = f"{c_name}.{s_name}.00000"
    record_path = os.path.join(RECORDS_DIR, g_name, s_name, r_name)
    ra = RecordAnalyzer(record_path)
    ra.analyze()
    for v in ra.get_results():
        related_data = adc.routing
        main_type = v[0]
        sub_type = v[1]
        if main_type == 'collision':
            for npc in npcs:
                if npc.nid == sub_type:
                    related_data = g_name + s_name + str(round(v[2]*0.1,1))
                    break
                else: continue

        if ViolationTracker.get_instance().add_violation(
                gname=g_name,
                sname=s_name,
                record_file=record_path,
                mt=main_type,
                st=sub_type,
                data=related_data,
        ):
            print(main_type, sub_type)


    minDistance = runner.get_min_distance()
    styleCollocation = ind.get_style_collocation()
    return minDistance, npcRobustness, styleCollocation


def cx_npc_section(ind1: NPCSection, ind2: NPCSection):
    index = randint(0, len(ind1.npcs) - 1)
    cx1 = ind1.npcs[index]
    cx2 = ind2.npcs[index]

    ind2.npcs[index] = cx1.get_same_one(cx1.routing,cx1.start_s,cx1.dest_s,cx1.start_t,cx1.style_list,cx1.id)
    ind1.npcs[index] = cx2.get_same_one(cx2.routing,cx2.start_s,cx2.dest_s,cx2.start_t,cx2.style_list,cx2.id)
    ind1.adjust_time()
    ind2.adjust_time()
    ind1 = ind1.init_self()
    ind2 = ind2.init_self()
    return ind2, ind1


def cx_scenario(ind1: Scenario, ind2: Scenario):

    ind1.npc_section, ind2.npc_section = cx_npc_section(
        ind1.npc_section, ind2.npc_section
    )
    return ind1, ind2


def mut_scenario(ind:Scenario):
    ind.npc_section = mut_npc_section(ind.npc_section)
    return ind,

def mut_npc_section(ind:NPCSection):
    index = randint(0, len(ind.npcs)-1)
    routing = ind.npcs[index].routing
    original_npc = ind.npcs.pop(index)
    mut_counter = 0
    while True:
        if ind.add_agent_for_s(NPCAgent.get_one_for_routing(original_npc.nid, routing), adc_start):
            break
        mut_counter += 1
        if mut_counter == 5:
            ind.add_agent_for_s(original_npc, adc_start)
            pass
    ind.adjust_time()
    ind = ind.init_self()
    return ind



def main():
    mp = MapParser(HD_MAP_PATH)
    model = NPCModel()
    container = ApolloContainer(APOLLO_ROOT, f'ROUTE_0')
    container.start_instance()
    container.start_dreamview()
    print(f'Dreamview at http://{container.ip}:{container.port}')

    POP_SIZE = 10
    OFF_SIZE = 10
    CXPB = 0.8
    MUTPB = 0.2

    toolbox = base.Toolbox()
    toolbox.register("evaluate", eval_scenario)
    toolbox.register("mate", cx_scenario)
    toolbox.register("mutate", mut_scenario)
    toolbox.register("select", tools.selNSGA2)

    srunner = ScenarioRunner(container)
    vt = ViolationTracker()

    start_time = datetime.now()

    population = [Scenario.get_conflict_one_S3() for _ in range(POP_SIZE)]
    for index, c in enumerate(population):
        c.gid = 0
        c.cid = index
    hof = tools.ParetoFront()

    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    hof.update(population)

    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("max", np.max, axis=0)
    stats.register("min", np.min, axis=0)
    logbook = tools.Logbook()
    logbook.header = 'gen', 'avg', 'max', 'min'

    curr_gen = 0
    while True:
        curr_gen += 1

        print(f'Generation {curr_gen}')
        print("start a generation")
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

        print("begin population generated")
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        for s in population:
            s.npc_section = s.npc_section.init_self()


        offspring = algorithms.varOr(
            population, toolbox, OFF_SIZE, CXPB, MUTPB)

        for index, c in enumerate(offspring):
            c.gid = curr_gen
            c.cid = index


        print("end population generated")
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        hof.update(offspring)

        population[:] = toolbox.select(population + offspring, POP_SIZE)

        record = stats.compile(population)
        logbook.record(gen=curr_gen, **record)
        print(logbook.stream)

        vt.save_to_file()
        with open('./data/log.bin', 'wb') as fp:
            pickle.dump(logbook, fp)


        print("end a generation")
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        curr_time = datetime.now()
        tdelta = (curr_time - start_time).total_seconds()
        print("time" + str(tdelta/3600))
        if tdelta / 3600 > 12:
            break

if __name__ == '__main__':
    main()
