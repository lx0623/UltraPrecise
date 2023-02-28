#ifndef ARIES_SERVER_SESSION
#define ARIES_SERVER_SESSION

#include <string>
#include <memory>
#include "DatabaseSchema.h"

namespace aries {

class AriesServerSession {
public:
    AriesServerSession() {}

    DatabaseSchemaPointer session_schema = nullptr;

};


typedef std::shared_ptr<AriesServerSession> AriesServerSessionPointer;

}//ns

#endif
