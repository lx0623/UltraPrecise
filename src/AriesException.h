//
// Created by tengjp on 19-8-14.
//

#ifndef AIRES_ARIESEXCEPTION_H
#define AIRES_ARIESEXCEPTION_H

#include <string>
#include <exception>
#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>
using std::string;
#define ER_FAKE_IMPL_OK 100
#define ER_FAKE_IMPL_EOF 101
namespace aries {
    // base exception class
    class AriesException : public std::exception {
    public:
        AriesException(int errCodeArg, const string& errMsgArg, const std::string& file_name = std::string(), int lineno = 0) :
                errCode(errCodeArg), errMsg(errMsgArg), file_name(file_name), lineno(lineno) {

        }
        int errCode;
        string errMsg;

        std::string file_name;
        int lineno;
    };

    // fake implementation for some query
    class AriesFakeImplException : public AriesException {
    public:
        AriesFakeImplException( int errCodeArg, const string &errMsgArg ) :
                AriesException(errCodeArg, errMsgArg)
        {
        }
    };

// #define AriesFrontendException(code, msg) AriesFrontendException(code, msg, __FILE__, __LINE__)
    struct cuda_exception_t: std::exception
    {
        cudaError_t result;

        cuda_exception_t( cudaError_t result_ )
                : result( result_ )
        {
        }
        virtual const char* what() const noexcept
        {
            return cudaGetErrorString( result );
        }
    };
    struct cu_exception_t: std::exception
    {
        CUresult result;

        cu_exception_t( CUresult result_ )
                : result( result_ )
        {
        }
        virtual const char* what() const noexcept
        {
            const char *msg;
            cuGetErrorString( result, &msg );
            return msg;
        }
    };
    struct nvrtc_exception_t: std::exception
    {
        nvrtcResult result;

        nvrtc_exception_t( nvrtcResult result_ )
                : result( result_ )
        {
        }
        virtual const char* what() const noexcept
        {
            return nvrtcGetErrorString( result );
        }
    };
    void ThrowNotSupportedException(const string& msg);
    void ThrowFakeImplException(int errCode, const string &msg);

    int FormatOutOfRangeValueError(const string& colName,
                                   int64_t lineIdx,
                                   string &errorMsg);
    int FormatTruncWrongValueError(const string &colName,
                                   const string &colValue,
                                   int64_t lineIdx,
                                   const char *valueTypeStr,
                                   string &errorMsg);
    int FormatDataTruncError(const string& colName,
                             int64_t lineIdx,
                             string &errorMsg);
    int FormatDataTooLongError( const string& colName,
                                int64_t lineIdx,
                                string &errorMsg);

    #define CheckTruncError(colName, typeStr, colStr, lineIdx, errorMsg)                           \
        if (*tail != '\0')                                                                          \
        {                                                                                           \
            int tmpErrCode;                                                                         \
            if (tail == colStr.c_str())                                                                \
                tmpErrCode = FormatTruncWrongValueError(colName, colStr, lineIdx, typeStr, errorMsg); \
            else                                                                                    \
                tmpErrCode = FormatDataTruncError(colName, lineIdx, errorMsg);                     \
            if (STRICT_MODE)                                                                        \
                return tmpErrCode;                                                                  \
            LOG( WARNING ) << "Convert data warning: " << errorMsg;                                   \
        }
    #define CheckCharLen( colName, len )                          \
    {                                                         \
        if ( len > ARIES_MAX_CHAR_WIDTH )                     \
        {                                                     \
            string msg( "Column length too big for column '"); \
            msg.append( colName ).append( "' (max = ").append( std::to_string( ARIES_MAX_CHAR_WIDTH ) ).append( ")" ); \
            ARIES_EXCEPTION_SIMPLE( ER_TOO_BIG_FIELDLENGTH, msg.data() ); \
        } \
    }
}

#endif //AIRES_ARIESEXCEPTION_H
