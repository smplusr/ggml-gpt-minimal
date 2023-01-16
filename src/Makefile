CXX	:= g++
CFLAGS	:= -Os -DGPT_2_117M #-DDEBUG

SOURCES	:= include.o utils.o ggml.o main.o
TARGET	:= main.elf



all:	$(TARGET)


%.o:	%.cpp
	$(CXX) $(CFLAGS) -c $<


%.elf:	$(SOURCES)
	$(CXX) -o $@ $^
