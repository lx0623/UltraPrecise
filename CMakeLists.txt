CMAKE_MINIMUM_REQUIRED(VERSION 3.14)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
PROJECT(Aires LANGUAGES CUDA CXX ASM)
include(CheckLanguage)
check_language(CUDA)

set(BINARY_NAME "rateup")

option(ENABLE_INIT_TABLE_CACHE "cache initial tables" OFF)
option(INIT_TABLE_TEST "build with initial table test flag" OFF)
option(BUILD_TEST "build with test" OFF)
option(BUILD_PROFILE "build with profile" ON)


include_directories(/usr/local/include src /usr/local/cuda/include)
include_directories(src/datatypes src/CudaAcc/algorithm src/server)
#include_directories(src/datatypes src/CudaAcc/algorithm src/server src/3rdparty/tcmalloc src/3rdparty/abseil-cpp src/3rdparty/benchmark/include)
include_directories(${CMAKE_BINARY_DIR})

link_directories(/usr/local/lib)

add_definitions(-DBUILD_WITH_CMAKE)

if(ENABLE_INIT_TABLE_CACHE)
    message("build with initial table cache")
    add_definitions("-DARIES_INIT_TABLE_CACHE")
else()
    message("build without initial table cache")
endif()

if(INIT_TABLE_TEST)
    message("build with initial table test flag")
    add_definitions("-DARIES_INIT_TABLE_TEST")
else()
    message("build without initial table test flag")
endif()

if(CMAKE_BUILD_TYPE MATCHES debug)
    SET(BUILD_RELEASE FALSE)
    message("will build debug")
elseif(CMAKE_BUILD_TYPE MATCHES Debug)
    SET(BUILD_RELEASE FALSE)
    message("will build debug")
elseif(CMAKE_BUILD_TYPE MATCHES DEBUG)
    SET(BUILD_RELEASE FALSE)
    message("will build debug")
else()
    SET(BUILD_RELEASE TRUE)
    message("will build release")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wreturn-type")
if(BUILD_RELEASE)
    add_definitions(-DNDEBUG)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -static-libstdc++")# -static-libgcc -static")
endif()

if(BUILD_PROFILE)
    message("will build with profile")
    add_definitions("-DARIES_PROFILE") 
else()
    message("will build without profile")
endif()

set(CMAKE_CXX_STANDARD 17)

set(SCHEMA_SOURCES
        src/schema/Schema.cpp
        src/schema/DBEntry.cpp
        src/schema/ColumnEntry.cpp
        src/schema/TableEntry.cpp
        src/schema/DatabaseEntry.cpp
        src/schema/SchemaManager.cpp
        src/schema/SchemaManager.h)

file(GLOB DATATYPES_SOURCES_CU "src/datatypes/*.cu")
set(ARIES_LIB_SOURCES
        ${DATATYPES_SOURCES_CU}
        src/AriesColumnDataIterator.cu
        src/AriesDataType.cu
        src/AriesColumnType.cu
        )

set(ARIES_LIB_HEADERS
        src/AriesDataType.h
        src/AriesColumnType.h
        src/AriesDefinition.h
        src/AriesColumnDataIterator.hxx
        src/CudaAcc/algorithm/cpptraits.hxx
        src/CudaAcc/algorithm/aries_char.hxx
        src/datatypes/aries_types.hxx
        src/datatypes/functions.hxx
        # src/datatypes/newDecimal.hxx
        src/datatypes/decimal.hxx
        src/datatypes/AriesDecimal.hxx
        src/datatypes/AriesIntervalTime.hxx
        src/datatypes/AriesDatetime.hxx
        src/datatypes/AriesDate.hxx
        src/datatypes/AriesTimeCalc.hxx
        src/datatypes/AriesTimestamp.hxx
        src/datatypes/AriesYear.hxx
        src/datatypes/AriesInnerHelper.hxx
        src/datatypes/AriesTime.hxx
        src/datatypes/AriesSqlFunctions.hxx
        src/datatypes/AriesCastFunctions.hxx
        src/datatypes/AriesDateFormat.hxx
        src/datatypes/AriesDataTypeUtil.hxx
        src/datatypes/AriesTruncateFunctions.hxx)

set(SQL_EXECUTOR_SOURCES
        src/frontend/SQLExecutor.cpp)

set(COMMON_SOURCES
    src/AriesException.cpp
    )

aux_source_directory(src/frontend FRONTEND_SOURCES)
list(REMOVE_ITEM FRONTEND_SOURCES
        SQL_EXECUTOR_SOURCES)

set(FRONTEND_FOR_PARSERV2 ${FRONTEND_SOURCES}  src/utils/string_util.cpp)

aux_source_directory(src/parserv2 PARSERV2_SOURCES)
aux_source_directory(src/optimizer OPTIMIZER_SOURCES)

list(REMOVE_ITEM FRONTEND_FOR_PARSERV2 
        src/utils/mutex_lock.cpp)


aux_source_directory(src/server/mysql MYSQL_SOURCES)
aux_source_directory(src/AriesEngine ARIESENGINE_SOURCES)
aux_source_directory(src/AriesEngine/transaction ARIESTRANSACTION_SOURCES)
aux_source_directory(src/AriesEngine/index ARIESINDEX_SOURCES)
# aux_source_directory(src/AriesEngine/index/btree ARIESBTREE_SOURCES)
aux_source_directory(src/Compression/dict ARIESCOMPRESSION_SOURCES)
aux_source_directory(src/AriesEngineWrapper AriesENGINEWRAPPER_SOURCES)
#aux_source_directory(src/RocksDB ARIESROCKSDB_SOURCES)
#aux_source_directory(src/RocksDB/myrocks ARIESMYROCKS_SOURCES)
aux_source_directory(src/CudaAcc CUDAACC_SOURCES)
aux_source_directory(src/CudaAcc/algorithm CUDAACC_ALGORITHM_SOURCES)
# aux_source_directory(src/datatypes DATATYPES_SOURCES)
aux_source_directory(src/utils ARIESUTILS_SOURCES)

list(REMOVE_ITEM ARIESENGINE_SOURCES
        src/AriesEngine/AriesScanNode.cpp)

# third parties
# file(GLOB_RECURSE TCMALLOC_SOURCES "src/3rdparty/tcmalloc/tcmalloc/*.cc")
# list(APPEND TCMALLOC_SOURCES "src/3rdparty/tcmalloc/tcmalloc/internal/percpu_rseq_x86_64.S")
# file(GLOB_RECURSE ABSEIL_SOURCES "src/3rdparty/abseil-cpp/absl/*.cc")

file(GLOB DATATYPES_SOURCES_CPP "src/datatypes/*.cpp")

if(BUILD_TEST)
    message("will build with test")
    add_definitions(-DBUILD_TEST)
    set(TEST_SOURCES
    #         test/testcases/datatypes/test_datatype_convert.cpp
    #         test/testcases/functions/test_cast.cpp
            #   test/testcases/datatypes/test_decimal.cu
     #         //0803
            #   test/testcases/CudaAcc/test_engine.cu
            # test/testcases/CudaAcc/test_pre_of_two_calc.cu
            # test/testcases/CudaAcc/test_pre_of_noncompact.cu
            # test/testcases/CudaAcc/test_pre_of_compact.cu
            # test/testcases/CudaAcc/test_pre_of_compact_trans.cu
            # test/testcases/CudaAcc/test_per_of_align_compact.cu
            # test/testcases/CudaAcc/test_common.cu
    #         # test/testcases/datatypes/test_newdecimal.cu
    #         test/testcases/datatypes/test_ariesdatetime.cu
    #         test/TestTpch_1.cpp
    #         test/TestTpch218_1.cpp
    #         test/TestTpch218_100.cpp
    #         test/TestTpchWithNull.cpp
    #         test/TestTpchWithWch.cpp
    #         test/TestCommonBase.cpp
    #         test/TestCommonBase.h
    #         test/TestPie.cpp
    #         test/TestTpch_100.cpp
    #         test/TestTpch_20.cpp
    #         test/TestBugFix.cpp
    #         test/TestTransaction.cpp
    #         test/TestUtils.cpp
    #         # src/server/InitDatabase.cpp
    #         test/testcases/AriesEngine/transaction/AriesMvccTestDataGenerator.cpp
    #         test/testcases/AriesEngine/transaction/TestAriesMvccTable.cpp
    #         # test/testcases/AriesEngine/transaction/TestAriesMvccTableCache.cpp
    #         test/testcases/AriesEngine/transaction/TestAriesTuple.cpp
    #         test/testcases/AriesEngine/transaction/TestAriesDeltaTable.cpp
    #         test/testcases/AriesEngine/transaction/TestAriesXLogManager.cpp
    #         test/testcases/AriesEngine/transaction/TestAriesXLogRecoveryer.cpp
    #         test/testcases/AriesEngine/TestAriesUpdateNode.cpp
    #         test/testcases/AriesEngine/TestAriesUpdateCalcNode.cpp
    #         test/testcases/AriesEngine/TestAriesConstantNode.cpp
    #         test/testcases/AriesEngine/TestAriesExchangeNode.cpp
    #         test/testcases/AriesEngine/AriesConstantGenerator.cpp
    #         test/testcases/AriesEngine/TestAriesInsertNode.cpp
    #         test/testcases/AriesEngine/TestAriesDeleteNode.cpp
    #         test/testcases/AriesEngine/TestInnerJoin.cpp
    #         test/testcases/AriesEngine/TestLeftJoin.cpp
    #         test/testcases/AriesEngine/TestRightJoin.cpp
    #         test/testcases/AriesEngine/TestFullJoin.cpp
    #         test/testcases/AriesEngine/QueryOpNode.cpp
    #         test/testcases/loaddata/test_loaddata.cpp
    #         test/testcases/Partition/TestPartition.cpp
    #         test/testcases/AriesEngine/transaction/TestAriesInitialTable.cpp
    #         test/testcases/Server/transaction/test_mysql_tx_proto.cpp
    #         # test/testcases/Compression/dict/TestAriesDictionary.cpp
    #         test/testcases/Compression/dict/TestAriesDictEncode.cpp
    # #        test/testcases/Compression/dict/TestAriesDictPerformance.cu
    #         test/testcases/Schema/TestSchema.cpp
    #         # test/testcases/AriesEngine/TestAriesDictScanNode.cpp
    #         # test/testcases/CudaAcc/TestAriesDecimalAlgorithm.cpp
    #         test/testcases/Optimizer/TestGroupbyOptimize.cpp
    #         test/testcases/Optimizer/TestSubqueryUnnesting.cpp
    #         test/testcases/Optimizer/TestSelfJoin.cpp
    #         #test/testcases/AriesEngine/index/btree/btree_test.cpp
    #         test/testcases/AriesEngine/TestJoin.cpp
            )
else()
    message("will build without test")
    set(TEST_SOURCES "")
endif()

find_file(CONFIG_PATH ".build.config.cmake"
          PATHS .)

set_property(GLOBAL PROPERTY CUDA_API_LEVEL "61")
if (NOT ${CONFIG_PATH} STREQUAL "CONFIG_PATH-NOTFOUND")
    include(".build.config.cmake")
endif()

get_property(CUDA_API_LEVEL GLOBAL PROPERTY CUDA_API_LEVEL)
if(NOT ${CUDA_API_LEVEL} MATCHES "7.*")
    add_definitions(-DLOWER_API_LEVEL)
endif()
message("cuda api level will be: " ${CUDA_API_LEVEL})

if(BUILD_RELEASE)
    set(CMAKE_CUDA_FLAGS "-DNDEBUG --default-stream per-thread --expt-extended-lambda --expt-relaxed-constexpr --relocatable-device-code true -gencode arch=compute_${CUDA_API_LEVEL},code=sm_${CUDA_API_LEVEL} -O3")
else(BUILD_RELEASE)
    set(CMAKE_CUDA_FLAGS "--default-stream per-thread --expt-extended-lambda --expt-relaxed-constexpr --relocatable-device-code=true -gencode arch=compute_${CUDA_API_LEVEL},code=sm_${CUDA_API_LEVEL}")
endif(BUILD_RELEASE)
set(CUDA_CUDART_LIBRARY "/usr/local/cuda/lib64/libcudart.so")

add_library(utils ${ARIESUTILS_SOURCES})
add_library(mysqlproto ${SQL_EXECUTOR_SOURCES} ${MYSQL_SOURCES})
add_library(schema ${SCHEMA_SOURCES})
add_library(frontend ${FRONTEND_SOURCES} ${PARSERV2_SOURCES} ${COMMON_SOURCES} ${OPTIMIZER_SOURCES} version_hash.h)
add_library(ariesdatatype ${ARIES_LIB_SOURCES})
add_library(accelerator
        ${CUDAACC_SOURCES}
        ${CUDAACC_ALGORITHM_SOURCES}
        ${DATATYPES_SOURCES_CPP}
        )
    # add_library(tcmalloc ${TCMALLOC_SOURCES} ${ABSEIL_SOURCES})

target_link_libraries(accelerator ariesdatatype)
target_link_libraries(mysqlproto libcrypto.a libboost_filesystem.a)
target_link_libraries(frontend mysqlproto utils)

add_executable(${BINARY_NAME}
        src/server/InitDatabase.cpp
        src/server/main.cc
        src/server/Configuration.cpp
        src/AriesAssert.cc
        src/AriesDeviceProperty.cpp
        src/AriesDeltaTableProperty.h
        ${TEST_SOURCES}
        ${AESUPPORTV2_SOURCES}
        ${ACCELERATOR_SOURCES}
        ${ARIESENGINE_SOURCES}
        ${ARIESTRANSACTION_SOURCES}
        ${ARIESCOMPRESSION_SOURCES}
        ${ARIESINDEX_SOURCES}
        # ${ARIESBTREE_SOURCES}
#        ${ARIESROCKSDB_SOURCES}
        ${ARIESMYROCKS_SOURCES}
        ${AriesENGINEWRAPPER_SOURCES}
        version_hash.h
        ${COMMON_SOURCES}
        )

add_custom_command(TARGET utils PRE_BUILD
    COMMAND cp  ../lib/3rdparty/libtcmalloc_minimal.a ${CMAKE_BINARY_DIR}
    )

# 这里声明一个假命令，fake_file 始终不存在会导致依赖它的 version_hash.h 在每次构建的时候都会自动生成
add_custom_command(
    OUTPUT fake_file
    COMMAND echo ""
)

# 生成 version_hash.h 文件，该文件会定义一个代表当前编译的 git commit 的宏
add_custom_command(
    OUTPUT version_hash.h
    COMMAND sh  ${PROJECT_SOURCE_DIR}/genversion.sh
    DEPENDS fake_file
)

add_custom_command(TARGET ${BINARY_NAME} POST_BUILD
    COMMAND cp ../src/server/mysql/share/english/errmsg.sys ./
)

if (BUILD_TEST)
    add_custom_command(TARGET ${BINARY_NAME} POST_BUILD
        # COMMAND cp -r ../test_tpch_queries ./
        # COMMAND cp -r ../smoke_test_queries ./
        # COMMAND cp -r ../test_tpch218_queries ./
        # COMMAND cp -r ../test_pie_queries ./
        # COMMAND cp -r ../test_resources ./
        # COMMAND cp ../test/scripts/* ./
        # COMMAND mkdir ./output
        # COMMAND mkdir ./output_format
        # COMMAND cp ../test_tpch_script/tpch_query.py ./
        # COMMAND cp ../test_tpch_script/format.py ./output
        # COMMAND cp ../test_tpch_script/varify.py ./output_format
        )
endif()

target_link_libraries(${BINARY_NAME} 
    # ${CMAKE_BINARY_DIR}/libtcmalloc_minimal.a
        accelerator 
        frontend 
        schema
        cuda
        nvrtc 
        dl 
        pthread
        # boost_filesystem
        unwind
        )
target_link_libraries(${BINARY_NAME}
        libgtest.a
        libgtest_main.a
        libgmock_main.a
        libboost_system.a
        libboost_filesystem.a
        libboost_serialization.a
        libglog.a
        libgflags.a
        # libunwind.a
        liblzma.a
        )

set_property(TARGET ${BINARY_NAME} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET ${BINARY_NAME} PROPERTY CUDA_ARCHITECTURES  ${CUDA_API_LEVEL})
set_property(TARGET ariesdatatype PROPERTY CUDA_ARCHITECTURES  ${CUDA_API_LEVEL} )
set_property(TARGET accelerator PROPERTY CUDA_ARCHITECTURES  ${CUDA_API_LEVEL} )
set_property(TARGET utils PROPERTY CUDA_ARCHITECTURES  ${CUDA_API_LEVEL} )

add_custom_command(TARGET ${BINARY_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/include
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/lib
    COMMAND cp ${CMAKE_BINARY_DIR}/libariesdatatype.a ${CMAKE_BINARY_DIR}/lib
    COMMAND cp ${ARIES_LIB_HEADERS} ${CMAKE_BINARY_DIR}/include
    WORKING_DIRECTORY ..
)

install(TARGETS ${BINARY_NAME} DESTINATION /usr/sbin)
install(FILES src/server/mysql/share/english/errmsg.sys DESTINATION /usr/sbin)
install(TARGETS ariesdatatype DESTINATION /usr/sbin/lib)
install(FILES ${ARIES_LIB_HEADERS} DESTINATION /usr/sbin/include/)



