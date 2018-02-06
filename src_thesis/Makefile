# Modified from the RL-Glue C/C++ codec sharedLibraryInterfaceExample's makefile.

# This will likely need to be changed to suit your installation. 
# Original: ALE := /home/daniel/Descargas/ale_0.4.4/ale_0_4
ALE := ..

DEBUG := 1

FLAGS := -I$(ALE)/src -I$(ALE)/src/controllers -I$(ALE)/src/os_dependent -I$(ALE)/src/environment -I$(ALE)/src/external -L$(ALE) # -D__USE_SDL
CXX := g++ -std=c++11
FILE := main
LDFLAGS := -lale -lz -lSDL -lSDL_gfx -lSDL_image

ifeq ($(strip $(DEBUG)), 1)
  FLAGS +=  -D__DEBUG
endif

all: main 

main: 
	$(CXX) -g $(FLAGS) $(FILE).cpp AgentSettings.cpp DynaAgent.cpp DynaMemories.cpp FunctionApproximationUtils.cpp LFAMethod.cpp PlayerAgent.cpp RandomAgent.cpp RAMAgent.cpp RAMIncrementalAgent.cpp Sarsa.cpp SearchAgent.cpp SingleActionAgent.cpp UCTNode.cpp UCTSearch.cpp $(LDFLAGS) -o $(FILE)

clean:
	rm -rf $(FILE) *.o

