//
// Created by 胡胜刚 on 2019-06-14.
//
#pragma once

#include <string>
#include <memory>

#include "Schema.h"


namespace aries {
namespace schema {

class SchemaManager {
private:
    std::shared_ptr<Schema> schema;

    explicit SchemaManager();

public:
    static SchemaManager* GetInstance();
    bool Load();
    bool LoadBaseSchama();

    bool InitSchema();
    Schema* GetSchema();
};

} // namespace schema
} // namespace aries

