TARGET = main.out
SRC = $(wildcard src/*.cpp)
OBJ = $(patsubst src/%.cpp, obj/%.o, $(SRC))


.PHONY: run clean

$(TARGET): $(OBJ)
	@echo Linking: $(OBJ)
	@g++ $(OBJ) -o $(TARGET)

$(OBJ): obj/%.o : src/%.cpp
	@echo Compiling: $< '=>' $@
	@g++ -Isrc -c -std=c++11 $< -o $@

run: $(TARGET)
	@echo Running: $(TARGET)
	@./$(TARGET)

clean:
	@echo Cleaning: $(OBJ)
	@-rm -f $(OBJ)
