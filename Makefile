CC = g++
CFLAGS = -Wall -Wextra -std=c++11 -Iinclude
LDFLAGS = -L/usr/local/lib
LIBS = -lOpenCL

SRCDIR = src
BUILDDIR = build
BINDIR = bin
KERNELSDIR = kernels
INCLUDEDIR = include

SOURCES := $(wildcard $(SRCDIR)/*.cpp)
OBJECTS := $(patsubst $(SRCDIR)/%,$(BUILDDIR)/%,$(SOURCES:.cpp=.o))
EXECUTABLE = $(BINDIR)/main

.PHONY: all clean run

all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	@mkdir -p $(BINDIR)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS) $(LIBS)

$(BUILDDIR)/%.o: $(SRCDIR)/%.cpp
	@mkdir -p $(BUILDDIR)
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -rf $(BUILDDIR)/* $(BINDIR)/*

run: $(EXECUTABLE)
	./$(EXECUTABLE)
