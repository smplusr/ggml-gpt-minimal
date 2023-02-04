MODEL	?= $(shell /bin/sh -c 'read -p "Enter a GPT-2 model (valid models are GPT_2_117M, GPT_2_345M, GPT_2_774M or GPT_2_1558M): " model; echo $$model')
DEBUG	?= $(shell /bin/sh -c 'read -p "Enable debbuging? [Y/n]" debug_enable; if [[ $$debug_enable = "Y" || $$debug_enable = "y" ]]; then echo -DDEBUG; fi')


CC	:= gcc -static -lpthread -pthread
CXX	:= g++ -static -lpthread -pthread
CFLAGS	:= -Os -D$(MODEL) $(DEBUG)

SOURCES	:= include.o utils.o ggml.o main.o
TARGET	:= main.elf



all:	$(TARGET)

clean:
	$(RM) *.elf

%.o:	%.cpp
	$(CXX) $(CFLAGS) -c $<


%.elf:	$(SOURCES)
	$(CXX) -o $@ $^
