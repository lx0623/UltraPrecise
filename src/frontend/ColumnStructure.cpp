#include "ColumnStructure.h"

namespace aries {
// ColumnValueType ColumnStructure::GetColumnValueTypeFromString(std::string arg_str) {
//
//     if (arg_str == "TEXT") return BiaodashiValueType::TEXT;
//     if (arg_str == "INTEGER") return BiaodashiValueType::INT;
//     if (arg_str == "DECIMAL") return BiaodashiValueType::FLOAT;
//
//     aries::TroubleHandler::throwWithTrace(std::invalid_argument(std::string("unregonized column type:") + arg_str));
//     return BiaodashiValueType::UNKNOWN;
// }

ColumnStructure::ColumnStructure(std::string arg_name, ColumnValueType arg_type, int arg_length, bool arg_allow_null,
                                 bool arg_is_primary) {

    assert(arg_name.empty() == false);

    this->name = arg_name;
    this->type = arg_type;
    this->length = arg_length;
    this->allow_null = arg_allow_null;
    this->is_primary = arg_is_primary;

}

std::string ColumnStructure::GetName() {

    return this->name;

}

ColumnValueType ColumnStructure::GetValueType() {
    return this->type;
}

bool ColumnStructure::GetIsPrimary() {

    return this->is_primary;

}

bool ColumnStructure::IsNullable() {
    return allow_null;
}

int ColumnStructure::GetLength() {
    if( type == ColumnValueType::TEXT || type == ColumnValueType::BINARY || type == ColumnValueType::VARBINARY )
    {
        return length;
    }
    else
    {
        return 1;
    }
}


std::shared_ptr<ColumnStructure> ColumnStructure::CloneColumnStructureIntoPointer() {

    std::shared_ptr<ColumnStructure> result =
            std::make_shared<ColumnStructure>(this->name,
                                              this->type,
                                              this->length,
                                              this->allow_null,
                                              this->is_primary);

    return result;
}


std::string ColumnStructure::ToString() {

    return std::string("Column:(" + this->name + ", "
                       + std::to_string((int) this->type) + ", "
                       + std::to_string(this->length) + ", "
                       + std::to_string(this->allow_null) + ", "
                       + std::to_string(this->is_primary) + ")");

}

void ColumnStructure::ResetName(std::string arg_value) {
    this->name = arg_value;
}


//bool is_fk = false;
//std::string fk_str = "";

bool ColumnStructure::GetIsFk() {
    return this->is_fk;
}

void ColumnStructure::SetIsFk(bool arg_value) {
    this->is_fk = arg_value;
}

std::string ColumnStructure::GetFkStr() {
    return this->fk_str;
}

void ColumnStructure::SetFkStr(std::string arg_value) {
    this->fk_str = arg_value;
}

std::shared_ptr<ColumnStructure> ColumnStructure::GetFkColumn() {
    return this->fk_column;
}

void ColumnStructure::SetFkColumn(std::shared_ptr<ColumnStructure> arg_value) {
    this->fk_column = arg_value;
}

int ColumnStructure::GetPrecision() {
    return numeric_precision;
}

int ColumnStructure::GetScale() {
    return numeric_scale;
}

void ColumnStructure::SetPresision(int precision) {
    numeric_precision = precision;
}

void ColumnStructure::SetScale(int scale) {
    numeric_scale = scale;
}

void ColumnStructure::SetEncodeType( EncodeType arg_encode_type ) {
    encode_type = arg_encode_type;
}

EncodeType ColumnStructure::GetEncodeType() const {
    return encode_type;
}

void ColumnStructure::SetEncodedIndexType( ColumnValueType arg_index_type ) {
    encode_index_data_type = arg_index_type;
}

ColumnValueType ColumnStructure::GetEncodedIndexType() const {
    return encode_index_data_type;
}

}
