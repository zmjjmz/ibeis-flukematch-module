CC=g++
CFLAGS=-O3 -Wall -shared -fpic
LDFLAGS=-I/usr/include/eigen3
SOURCES=flukematch_lib.cpp
OBJECTS=$(SOURCES:.cpp=.so)
all:
	$(CC) $(LDFLAGS) $(SOURCES) -o $(OBJECTS) $(CFLAGS)
