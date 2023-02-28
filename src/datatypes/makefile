CC=nvcc
CUDA_PATH=/usr/local/cuda
CUDA_FLAGS=-I $(CUDA_PATH)/include -I. -I.. -I../CudaAcc/algorithm/ -L $(CUDA_PATH)/lib64 --std=c++14 --relocatable-device-code=true --expt-relaxed-constexpr -gencode arch=compute_75,code=compute_75
OBJS=decimal.o AriesDate.o AriesDatetime.o AriesIntervalTime.o AriesTime.o AriesTimestamp.o AriesYear.o AriesTimeCalc.o AriesInnerHelper.o AriesSqlFunctions.o AriesDataTypeUtil.o AriesDateFormat.o AriesDateTimeLocales.o AriesCastFunctions.o AriesTruncateFunctions.o functions.o 

%.o: %.cu
	$(CC) -c -o build/$@ $< $(CUDA_FLAGS)

AriesColumnDataIterator.o: ../AriesColumnDataIterator.hxx ../AriesColumnDataIterator.cu
	$(CC) -c ../AriesColumnDataIterator.cu -o build/AriesColumnDataIterator.o $(CUDA_FLAGS)
	
AriesDataType.o: ../AriesDataType.h ../AriesDataType.cu
	$(CC) -c ../AriesDataType.cu -o build/AriesDataType.o $(CUDA_FLAGS)
	
AriesColumnType.o: ../AriesColumnType.h ../AriesColumnType.cu
	$(CC) -c ../AriesColumnType.cu -o build/AriesColumnType.o $(CUDA_FLAGS)

all: libariesdatatype.a
	

libariesdatatype.a: $(OBJS)  AriesColumnDataIterator.o AriesDataType.o AriesColumnType.o
	nvcc -lib -o build/libariesdatatype.a build/AriesDataType.o build/AriesColumnType.o build/decimal.o build/AriesDate.o build/AriesDatetime.o build/AriesIntervalTime.o build/AriesTime.o build/AriesTimestamp.o build/AriesYear.o build/AriesTimeCalc.o build/AriesInnerHelper.o build/AriesSqlFunctions.o build/AriesDataTypeUtil.o build/AriesDateFormat.o build/AriesDateTimeLocales.o build/AriesCastFunctions.o build/AriesTruncateFunctions.o build/AriesColumnDataIterator.o build/functions.o

deploy: libariesdatatype.a
	cp build/libariesdatatype.a ../../lib/
	cp *.hxx ../../include/
	cp ../AriesDefinition.h ../../include/
	cp ../AriesColumnType.h ../../include/
	cp ../AriesDataType.h ../../include/
	cp ../CudaAcc/algorithm/cpptraits.hxx ../../include/
	cp ../CudaAcc/algorithm/aries_char.hxx ../../include/
	cp ../datatypes/functions.hxx ../../include/
	cp ../datatypes/aries_types.hxx ../../include/
	cp ../AriesColumnDataIterator.hxx ../../include/
.PHONY: clean

clean:
	rm -f -r build/*.o build/*.a
