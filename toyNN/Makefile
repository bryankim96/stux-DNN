TARGET_EXEC ?= a.out

BUILD_DIR ?= ./build
SRC_DIRS ?= ./src


CXX=g++

CXXFLAGS= -std=c++17 -O2

# JSON sources
# JSONSOURCES=src/JSON.cpp src/JSONValue.cpp
# HEADERS=src/JSON.h src/JSONValue.h

all: main

main: setup toyNN.o main.o
	$(CXX) $(CXXFLAGS) -o $(BUILD_DIR)/main $(BUILD_DIR)/main.o $(BUILD_DIR)/JSON.o $(BUILD_DIR)/JSONValue.o $(BUILD_DIR)/toyNN.o


main.o: setup toyNN.o
	$(CXX) $(CXXFLAGS) -o $(BUILD_DIR)/main.o -c $(SRC_DIRS)/main.cpp

toyNN.o: setup JSON
	$(CXX) $(CXXFLAGS) -o $(BUILD_DIR)/toyNN.o -c $(SRC_DIRS)/toyNN.cpp

JSON: setup
	$(CXX) $(CXXFLAGS) -o $(BUILD_DIR)/JSON.o -c $(SRC_DIRS)/JSON.cpp #$(BUILD_DIR)/JSONValue.o
	$(CXX) $(CXXFLAGS) -o $(BUILD_DIR)/JSONValue.o -c $(SRC_DIRS)/JSONValue.cpp

#JSONValue.o: setup
#	$(CXX) $(CXXFLAGS) -o $(BUILD_DIR)/JSONValue.o -c $(SRC_DIRS)/JSONValue.cpp

setup:
	$(MKDIR_P) $(BUILD_DIR)

.PHONY: clean

clean:
	$(RM) -r $(BUILD_DIR)

-include $(DEPS)

MKDIR_P ?= mkdir -p
