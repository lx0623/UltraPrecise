#ifndef ARIES_COLUMN_STRUCTURE
#define ARIES_COLUMN_STRUCTURE

#include <cassert>
#include <iostream>
#include <fstream>
#include <vector>
#include <memory>

#include "TroubleHandler.h"
#include "VariousEnum.h"


namespace aries {
typedef BiaodashiValueType ColumnValueType;

class ColumnStructure {


private:

    std::string name;
    ColumnValueType type;
    int length;
    bool allow_null;
    bool is_primary;

    bool is_fk = false;
    std::string fk_str = "";
    std::shared_ptr<ColumnStructure> fk_column = nullptr;

    ColumnStructure(const ColumnStructure &arg);

    ColumnStructure &operator=(const ColumnStructure &arg);

    int numeric_precision = -1;
    int numeric_scale = -1;

    EncodeType encode_type = EncodeType::NONE;
    ColumnValueType encode_index_data_type = ColumnValueType::UNKNOWN;

public:


    // static ColumnValueType GetColumnValueTypeFromString(std::string arg_str);

    ColumnStructure(std::string arg_name, ColumnValueType arg_type, int arg_length, bool arg_allow_null,
                    bool arg_is_primary);

    std::string GetName();


    ColumnValueType GetValueType();

    bool GetIsPrimary();

    bool IsNullable();

    int GetLength();


    std::shared_ptr<ColumnStructure> CloneColumnStructureIntoPointer();

    std::string ToString();

    void ResetName(std::string arg_value);


    //bool is_fk = false;
    //std::string fk_str = "";

    bool GetIsFk();

    void SetIsFk(bool arg_value);

    std::string GetFkStr();

    void SetFkStr(std::string arg_value);

    std::shared_ptr<ColumnStructure> GetFkColumn();

    void SetFkColumn(std::shared_ptr<ColumnStructure> arg_value);

    int GetPrecision();
    int GetScale();
    void SetPresision(int precision);
    void SetScale(int scale);

    void SetEncodeType( EncodeType arg_encode_type );
    EncodeType GetEncodeType() const;
    void SetEncodedIndexType( ColumnValueType arg_index_type );
    ColumnValueType GetEncodedIndexType() const;
};


typedef std::shared_ptr<ColumnStructure> ColumnStructurePointer;

}

#endif
