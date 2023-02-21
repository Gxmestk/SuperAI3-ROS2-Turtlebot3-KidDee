#Defaults
NAME				= kidDeeMazeRunner
PY 					= python3

COLOUR_GREEN=\033[0;32m
COLOUR_RED=\033[0;31m
COLOUR_BLUE=\033[0;34m
COLOUR_YEL=\033[0;33m
COLOUR_END=\033[0m


#Make defaults

ex:					
					@echo "$(COLOUR_RED)training: make train ddpg 1$(COLOUR_END)"
					@echo "$(COLOUR_YEL)make train algo=ddpg mode=1$(COLOUR_END)"
					@echo "$(COLOUR_RED)testing: make test ddpg 0 'examples/ddpg_0' 8000 9$(COLOUR_END)"
					@echo "$(COLOUR_YEL)make algo=ddpg  mode=0 dir='examples/ddpg_0' ep=8000 stage=9$(COLOUR_END)"


train:				launch env gazebo agent

test:				launch env gazebo agent



env:
					$(PY) kidDeeEnv.py &
gazebo:
					$(PY) kidDeeEnv.py &
launch:
					$(PY) turtlebot3_drl_stage9.launch.py &
agent:
					$(PY) kidDeeAgent.py ${algo} ${mode} ${dir} ${ep} ${stage} &


clear_env:
					ps -ef | grep kidDeeEnv | grep -v grep | awk '{print $$2}' | xargs kill
clear_agent:
					ps -ef | grep kidDeeAgent | grep -v grep | awk '{print $$2}' | xargs kill

#ps -ef | awk 'kidDeeAgent/{print $2}' | xargs kill
clear_gazebo:
					ps -ef | grep kidDeeAgent | grep -v grep | awk '{print $$2}' | xargs kill
clear_launch:
					ps -ef | grep kidDeeAgent | grep -v grep | awk '{print $$2}' | xargs kill

clear:			clear_env ;clear_agent; clear_gazebo; clear_launch



#.PHONY
.PHONY:				test train code env gazebo launch agent clear_env clear_agent clear_gazebo clear_launch clear ex