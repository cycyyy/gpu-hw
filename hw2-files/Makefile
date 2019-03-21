
CC=nvcc -g
ARCH=-arch=sm_30
SOURCES=main.cu
OBJECTS=$(SOURCES:.cpp=.o)
EXECUTABLE=vector_add
all: $(SOURCES) $(EXECUTABLE)
    
$(EXECUTABLE): $(OBJECTS) 
	$(CC) $(ARCH) $(OBJECTS) -o $@
.PHONY : clean
clean :
	-rm $(EXECUTABLE)

