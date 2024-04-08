CC = gcc
CFLAGS = -Wall -Wextra -Iinclude -lOpenCL
LDFLAGS = -L/usr/local/lib
LIBS = -lOpenCL

SRCDIR = src
BUILDDIR = build
BINDIR = bin
KERNELSDIR = kernels
INCLUDEDIR = include

SOURCES := $(wildcard $(SRCDIR)/*.c)
OBJECTS := $(patsubst $(SRCDIR)/%,$(BUILDDIR)/%,$(SOURCES:.c=.o))
EXECUTABLE = $(BINDIR)/main

.PHONY: all clean run

all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	@mkdir -p $(BINDIR)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS) $(LIBS)

$(BUILDDIR)/%.o: $(SRCDIR)/%.c
	@mkdir -p $(BUILDDIR)
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -rf $(BUILDDIR)/* $(BINDIR)/*

run: $(EXECUTABLE)
	./$(EXECUTABLE)
