CC=gcc
NVCC=nvcc # cuda compiler
LIBS= 
SOURCE_DIR= .
BIN_DIR= .
CFLAGS= -O1 -g
LDFLAGS= -lm 
#compile the cuda program too
OBJS=$(SOURCE_DIR)/canny_edge.o $(SOURCE_DIR)/hysteresis.o $(SOURCE_DIR)/pgm_io.o $(SOURCE_DIR)/CudaCode.o
EXEC= canny
INCS= -I.
CSRCS= $(SOURCE_DIR)/canny_edge \
	$(SOURCE_DIR)/hysteresis.c \
	$(SOURCE_DIR)/pgm_io.c

# cuda source file
CUDA_SRC= $(SOURCE_DIR)/CudaCode.cu


PIC=pics/pic_large.pgm

all: canny

# links all the final obj files using nvcc 
canny: $(OBJS)
	$(NVCC) $(CFLAGS)  -o $@ $? $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# compile the cuda program into an object file
%.o: %.cu
	$(NVCC) $(CFLAGS) -c $< -o $@

run: $(EXEC) $(PIC)
	./$(EXEC) $(PIC) 2.5   0.25  0.5
# 			        sigma tlow  thigh

	

gprof:	CFLAGS +=  -pg
gprof:  LDFLAGS += -pg 
gprof:	clean all
	echo "./$(EXEC) $(PIC) 2.0 0.5 0.5" > lastrun.binary
	./$(EXEC) $(PIC) 2.0 0.5 0.5
	gprof -b ./$(EXEC) > gprof_$(EXEC).txt
	./run_gprof.sh canny


clean:
	@-rm -rf canny $(OBJS) gmon.out

.PHONY: clean comp exe run all