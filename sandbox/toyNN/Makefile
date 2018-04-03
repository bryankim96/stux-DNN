TARGET_EXEC ?= a.out

BUILD_DIR ?= ./build
SRC_DIRS ?= ./src


CXX=g++

CXXFLAGS= -std=c++17 -O2

all: main

main: setup toyNN.o main.o
	$(CXX) $(CXXFLAGS) -o $(BUILD_DIR)/main $(BUILD_DIR)/main.o $(BUILD_DIR)/toyNN.o


main.o: setup toyNN.o
	$(CXX) $(CXXFLAGS) -o $(BUILD_DIR)/main.o -c $(SRC_DIRS)/main.cpp

toyNN.o: setup
	$(CXX) $(CXXFLAGS) -o $(BUILD_DIR)/toyNN.o -c $(SRC_DIRS)/toyNN.cpp

setup:
	$(MKDIR_P) $(BUILD_DIR)

.PHONY: clean

clean:
	$(RM) -r $(BUILD_DIR)

-include $(DEPS)

MKDIR_P ?= mkdir -p
