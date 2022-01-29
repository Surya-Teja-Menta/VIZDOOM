
from cProfile import run
from train import *
from train.basic import *
from train.deadly_corridor import *
from train.defend_center import *
from train.defend_line import *
from train.health_gathering import *
from train.take_cover import *

def run_basic():
    # train basic()
    basic=ViZDoomGym()
    basic.train()
    basic.test()

def run_deadly_corridor():
    # train deadly_corridor()
    deadly_corridor=ViZDoomGym()
    deadly_corridor.train()
    deadly_corridor.test()

def run_defend_center():
    # train defend_center()
    defend_center=ViZDoomGym()
    defend_center.train()
    defend_center.test()

def run_defend_line():
    # train defend_line()
    defend_line=ViZDoomGym()
    defend_line.train()
    defend_line.test()

def run_health_gathering():
    # train health_gathering()
    health_gathering=ViZDoomGym()
    health_gathering.train()
    health_gathering.test()

def run_take_cover():
    # train take_cover()
    take_cover=ViZDoomGym()
    take_cover.train()
    take_cover.test()

#start Basic Level
run_basic()
#start Deadly Corridor Level
run_deadly_corridor()
#start Defend Center Level
run_defend_center()
#start Defend Line Level
run_defend_line()
#start Health Gathering Level
run_health_gathering()
#start Take Cover Level
run_take_cover()
