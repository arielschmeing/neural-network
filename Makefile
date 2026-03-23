CXX = g++
CXXFLAGS = -I/usr/local/Aria/include -I./include
CXXFLAGS += -Wno-deprecated-declarations
LDFLAGS = -L/usr/local/Aria/lib -lAria
SRC_DIR = src
OBJ_DIR = build
INCLUDE_DIR = include
SRC = $(wildcard $(SRC_DIR)/*.cpp)
OBJ = $(SRC:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)
TARGET = $(OBJ_DIR)/main

all: $(TARGET)

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

$(TARGET): $(OBJ)
	$(CXX) $(OBJ) -o $(TARGET) $(LDFLAGS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf $(OBJ_DIR)

.PHONY: all clean