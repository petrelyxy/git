
NVCC        = nvcc
NVCC_FLAGS  = -O3 -I/usr/local/cuda/include
LD_FLAGS    = -lcudart -L/usr/local/cuda/lib
EXE	        = stencil_student
OBJ	        = stencil_student.o

default: $(EXE)

stencil_student.o: stencil_student.cu
	$(NVCC) -c -o $@ stencil_student.cu $(NVCC_FLAGS)

$(EXE): $(OBJ)
	$(NVCC) $(OBJ) -o $(EXE) $(LD_FLAGS)

clean:
	rm -rf *.o $(EXE)

