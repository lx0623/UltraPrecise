//
// Created by 胡胜刚 on 2019-06-14.
//

#include "AriesAssert.h"

#include "SchemaManager.h"

namespace aries {
namespace schema {

SchemaManager::SchemaManager() {
    schema = std::make_shared<Schema>();
}

SchemaManager* SchemaManager::GetInstance() {
    static SchemaManager instance;

    return &instance;
}

bool SchemaManager::Load() {
    return schema->LoadSchema();
}

bool SchemaManager::LoadBaseSchama()
{
    return schema->LoadBaseSchema();
}

bool SchemaManager::InitSchema() {
    return schema->Init();
}

Schema* SchemaManager::GetSchema() {
    return schema.get();
}

}
}
