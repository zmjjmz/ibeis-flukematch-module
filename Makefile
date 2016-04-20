CC=g++
CFLAGS=-O3 -Wall -shared -fpic
# TODO: CMAKE
# -I opt/local/include/eigen3
LDFLAGS=-I/usr/include/eigen3
SOURCES=flukematch_lib.cpp
OBJECTS=$(SOURCES:.cpp=.so)
all:
	$(CC) $(LDFLAGS) $(SOURCES) -o $(OBJECTS) $(CFLAGS)
