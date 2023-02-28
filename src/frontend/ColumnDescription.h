#ifndef ARIES_COLUMN_DESCRIPTION
#define ARIES_COLUMN_DESCRIPTION

#include <string>
#include <memory>
#include "frontend/CommonBiaodashi.h"
#include <server/mysql/include/mysql_com.h>

namespace aries {
class PT_column_attr_base {
protected:
    PT_column_attr_base() {}
public:
    virtual void apply_type_flags(ulong *) const {}
    /**
     Check for the [NOT] ENFORCED characteristic.

     @returns true  if the [NOT] ENFORCED follows the CHECK(...) clause,
              false otherwise.
   */
    // virtual bool has_constraint_enforcement() const { return false; } // 8.0

    /**
      Check if constraint is enforced.
      Method must be called only when has_constraint_enforcement() is true (i.e
      when [NOT] ENFORCED follows the CHECK(...) clause).

      @returns true  if constraint is enforced.
               false otherwise.
    */
    // virtual bool is_constraint_enforced() const { return false; } // 8.0

};
using PT_column_attr_base_ptr = std::shared_ptr<PT_column_attr_base>;
using ColAttrList = std::shared_ptr<std::vector<PT_column_attr_base_ptr>>;
/**
  Node for the @SQL{NULL} column attribute

  @ingroup ptn_column_attrs
*/
    class PT_null_column_attr : public PT_column_attr_base {
    public:
        virtual void apply_type_flags(ulong *type_flags) const {
            *type_flags &= ~NOT_NULL_FLAG;
            *type_flags |= EXPLICIT_NULL_FLAG;
        }
    };
    /**
  Node for the @SQL{NOT NULL} column attribute

  @ingroup ptn_column_attrs
*/
    class PT_not_null_column_attr : public PT_column_attr_base {
        virtual void apply_type_flags(ulong *type_flags) const {
            *type_flags |= NOT_NULL_FLAG;
        }
    };
    /**
  Node for the @SQL{NOT SECONDARY} column attribute

  @ingroup ptn_column_attrs
*/
    class PT_secondary_column_attr : public PT_column_attr_base {
    public:
        void apply_type_flags(unsigned long *type_flags) const override {
            *type_flags |= NOT_SECONDARY_FLAG;
        }
    };

/**
  Node for the @SQL{UNIQUE [KEY]} column attribute

  @ingroup ptn_column_attrs
*/
    class PT_unique_key_column_attr : public PT_column_attr_base {
    public:
        virtual void apply_type_flags(ulong *type_flags) const {
            *type_flags |= UNIQUE_FLAG;
        }
    };

/**
  Node for the @SQL{PRIMARY [KEY]} column attribute

  @ingroup ptn_column_attrs
*/
    class PT_primary_key_column_attr : public PT_column_attr_base {
    public:
        virtual void apply_type_flags(ulong *type_flags) const {
            *type_flags |= PRI_KEY_FLAG | NOT_NULL_FLAG;
        }

        std::string name;
    };

/**
  Node for the @SQL{[CONSTRAINT [symbol]] CHECK '(' expr ')'} column attribute.

  @ingroup ptn_column_attrs
*/
    class PT_check_constraint_column_attr : public PT_column_attr_base {
    };

/**
  Node for the @SQL{[NOT] ENFORCED} column attribute.

  @ingroup ptn_column_attrs
*/
    class PT_constraint_enforcement_attr : public PT_column_attr_base {
    public:
        // explicit PT_constraint_enforcement_attr(bool enforced)
        //         : m_enforced(enforced) {}

        // bool has_constraint_enforcement() const override { return true; }

        // bool is_constraint_enforced() const override { return m_enforced; }

    private:
        // const bool m_enforced;
    };

/**
  Node for the @SQL{COMMENT @<comment@>} column attribute

  @ingroup ptn_column_attrs
*/
    class PT_comment_column_attr : public PT_column_attr_base {
        const std::string comment;

    public:
        explicit PT_comment_column_attr(const std::string &comment)
                : comment(comment) {}
    };

/**
  Node for the @SQL{COLLATE @<collation@>} column attribute

  @ingroup ptn_column_attrs
*/
    class PT_collate_column_attr : public PT_column_attr_base {
    };

// Specific to non-generated columns only:

/**
  Node for the @SQL{DEFAULT @<expression@>} column attribute

  @ingroup ptn_not_gcol_attr
*/
    class PT_default_column_attr : public PT_column_attr_base {
    public:
        BiaodashiPointer expr;
        explicit PT_default_column_attr( const BiaodashiPointer & argExpr) : expr(argExpr) {}
        // virtual void apply_default_value(Item **value) const { *value = item; }
        virtual void apply_type_flags(ulong *type_flags) const {
            auto commonExpr = std::dynamic_pointer_cast<CommonBiaodashi>( expr );
            if ( BiaodashiType::Null == commonExpr->GetType() ) *type_flags |= EXPLICIT_NULL_FLAG;
        }
    };
        /**
      Node for the @SQL{UPDATE NOW[([@<precision@>])]} column attribute

      @ingroup ptn_not_gcol_attr
    */
    class PT_on_update_column_attr : public PT_column_attr_base {
    };
    /**
  Node for the @SQL{AUTO_INCREMENT} column attribute

  @ingroup ptn_not_gcol_attr
*/
    class PT_auto_increment_column_attr : public PT_column_attr_base {
    public:
        virtual void apply_type_flags(ulong *type_flags) const {
            *type_flags |= AUTO_INCREMENT_FLAG | NOT_NULL_FLAG;
        }
    };
        /**
      Node for the @SQL{SERIAL DEFAULT VALUE} column attribute

      @ingroup ptn_not_gcol_attr
    */
    class PT_serial_default_value_column_attr : public PT_column_attr_base {
    public:
        virtual void apply_type_flags(ulong *type_flags) const {
            *type_flags |= AUTO_INCREMENT_FLAG | NOT_NULL_FLAG | UNIQUE_FLAG;
        }
    };
        /**
  Node for the @SQL{COLUMN_FORMAT @<DEFAULT|FIXED|DYNAMIC@>} column attribute

  @ingroup ptn_not_gcol_attr
*/
    class PT_column_format_column_attr : public PT_column_attr_base {
    public:
        // explicit PT_column_format_column_attr(column_format_type format)
        //         : format(format) {}

        virtual void apply_type_flags(ulong *type_flags) const {
            *type_flags &= ~(FIELD_FLAGS_COLUMN_FORMAT_MASK);
            // *type_flags |= format << FIELD_FLAGS_COLUMN_FORMAT;
        }
    };
            /**
  Node for the @SQL{STORAGE @<DEFAULT|DISK|MEMORY@>} column attribute

  @ingroup ptn_not_gcol_attr
*/
    class PT_storage_media_column_attr : public PT_column_attr_base {

    };
    /// Node for the SRID column attribute
    class PT_srid_column_attr : public PT_column_attr_base {

    };
    /// Node for the generated default value, column attribute
    class PT_generated_default_val_column_attr : public PT_column_attr_base {

    };
    class PT_encode_type_column_attr : public PT_column_attr_base {
    public:
        PT_encode_type_column_attr( aries::EncodeType arg_encode_type,
                                    const string& arg_index_data_type,
                                    const string& arg_dict_name )
            : encode_type( arg_encode_type ),
              index_data_type( arg_index_data_type ),
              dict_name( arg_dict_name ) {}
        aries::EncodeType encode_type;
        string index_data_type;
        std::string dict_name;
    };
enum keytype {
    KEYTYPE_PRIMARY,
    KEYTYPE_UNIQUE,
    KEYTYPE_MULTIPLE,
    KEYTYPE_FULLTEXT,
    KEYTYPE_SPATIAL,
    KEYTYPE_FOREIGN
};

class TableElementDescription {
public:
    virtual std::string ToString() =0;
    virtual bool IsColumnDesc() = 0;
};
using TableElementDescriptionPtr = std::shared_ptr<TableElementDescription>;
class PT_table_constraint_def: public TableElementDescription {
public:
    virtual std::string ToString() {
        return "";
    };
    virtual bool IsColumnDesc() { return false; }
};

class PT_table_key_constraint_def : public PT_table_constraint_def
{
public:
    PT_table_key_constraint_def( const std::string& name,
                                 const std::string& index_name,
                                 const keytype& type,
                                 const std::vector< std::string >& keys )
    {
        this->name = name;
        this->index_name = index_name;
        this->type = type;
        this->keys.assign( keys.cbegin(), keys.cend() );
    }

    PT_table_key_constraint_def( const std::string& name,
                                 const std::string& index_name,
                                 const keytype& type,
                                 const std::vector< std::string >& keys,
                                 const BasicRelPointer& referencedTable,
                                 const std::vector< std::string >& foreignkeys )
    {
        this->name = name;
        this->type = type;
        this->keys.assign( keys.cbegin(), keys.cend() );
        referenced_table = referencedTable;
        this->foreign_keys.assign( foreignkeys.cbegin(), foreignkeys.cend() );
    }

    keytype GetType() const
    {
        return type;
    }

    const std::vector< std::string >& GetKeys() const
    {
        return keys;
    }

    const std::vector< std::string >& GetForeignKeys() const
    {
        return foreign_keys;
    }

    BasicRelPointer GetReferencedTable() const
    {
        return referenced_table;
    }

    std::string GetName() const
    {
        return name;
    }

    std::string GetIndexName() const
    {
        return index_name;
    }

private:
    keytype type;
    BasicRelPointer referenced_table;
    std::vector< std::string > keys;
    std::vector< std::string > foreign_keys;
    std::string name;
    std::string index_name;
};

class PT_inline_index_definition : public PT_table_constraint_def {
    typedef PT_table_constraint_def super;

public:
    PT_inline_index_definition(keytype type_par, const std::string &name_arg
                               /*
                               PT_base_index_option *type,
                               List<PT_key_part_specification> *cols,
                               Index_options options
                               */)
            : m_keytype(type_par),
              m_name(name_arg)
              /*
              m_type(type),
              m_columns(cols),
              m_options(options)
               */ {}

private:
    keytype m_keytype;
    const std::string m_name;
    // PT_base_index_option *m_type;
    // List<PT_key_part_specification> *m_columns;
    // Index_options m_options;
};

class PT_foreign_key_definition : public PT_table_constraint_def {
    typedef PT_table_constraint_def super;

public:
    PT_foreign_key_definition(const std::string &constraint_name,
                              const std::string &key_name
                              /*
                              List<PT_key_part_specification> *columns,
                              Table_ident *referenced_table,
                              List<Key_part_spec> *ref_list,
                              fk_match_opt fk_match_option,
                              fk_option fk_update_opt, fk_option fk_delete_opt
                               */)
            : m_constraint_name(constraint_name),
              m_key_name(key_name)
              /*
              m_columns(columns),
              m_referenced_table(referenced_table),
              m_ref_list(ref_list),
              m_fk_match_option(fk_match_option),
              m_fk_update_opt(fk_update_opt),
              m_fk_delete_opt(fk_delete_opt)
               */
              {}


private:
    const std::string m_constraint_name;
    const std::string m_key_name;
    // List<PT_key_part_specification> *m_columns;
    // Table_ident *m_referenced_table;
    // List<Key_part_spec> *m_ref_list;
    // fk_match_opt m_fk_match_option;
    // fk_option m_fk_update_opt;
    // fk_option m_fk_delete_opt;
};
class ColumnDescription : public TableElementDescription {

public:
    ColumnDescription() {
    }
    virtual bool IsColumnDesc() { return true; }

    std::string column_name;

    std::string column_type;
    int column_major_len = -1;
    int column_minor_len = -1;

    int column_id;

    bool is_unsigned = false;
    bool not_null = false;
    bool explicit_nullable = false;
    bool has_default = false;
    bool explicit_default_null = false;
    bool primary_key = false;
    bool unique_key = false;
    bool multi_key = false;
    aries::EncodeType encode_type = aries::EncodeType::NONE;
    string encode_index_data_type;
    std::string dict_name;

    bool foreign_key = false;
    ulong type_flags = 0;
    std::string fk_table_name;
    std::string fk_column_name;
    std::string primary_key_name;
    std::shared_ptr<std::string> default_value;
    CommonBiaodashiPtr default_value_expr;
    void InitColumnAttr(const ColAttrList& col_attr_list ) {
        foreign_key = false;
        primary_key = false;
        not_null = false;
        if (col_attr_list) {
            for (auto columnAttr : *col_attr_list) {
                columnAttr->apply_type_flags(&type_flags);
                if ( PT_default_column_attr* attr = dynamic_cast<PT_default_column_attr*>( columnAttr.get() ) )
                {
                    has_default = true;
                    if ( type_flags & EXPLICIT_NULL_FLAG )
                        explicit_default_null = true;
                    default_value_expr = std::dynamic_pointer_cast<CommonBiaodashi > ( attr->expr );
                    if ( BiaodashiType::Null != default_value_expr->GetType() )
                        default_value = std::make_shared<std::string>( default_value_expr->ContentToString() );
                }
                else if ( PT_null_column_attr* attr = dynamic_cast< PT_null_column_attr* >( columnAttr.get() ) )
                {
                    ( void )attr;
                    explicit_nullable = true;
                }
                else if ( auto primary_key_attr = std::dynamic_pointer_cast< PT_primary_key_column_attr >( columnAttr ) )
                {
                    primary_key_name = primary_key_attr->name;
                }
                else if ( auto encode_type_attr = std::dynamic_pointer_cast< PT_encode_type_column_attr >( columnAttr ) )
                {
                    encode_type = encode_type_attr->encode_type;
                    encode_index_data_type = encode_type_attr->index_data_type;
                    dict_name = encode_type_attr->dict_name;
                }
            }
        }
        if (type_flags & UNSIGNED_FLAG) {
            is_unsigned = true;
        }
        if (type_flags & PRI_KEY_FLAG) {
            primary_key = true;
        }
        if (type_flags & UNIQUE_FLAG) {
            unique_key = true;
        }
        if (type_flags & MULTIPLE_KEY_FLAG) {
            multi_key = true;
        }
        if (type_flags & NOT_NULL_FLAG) {
            if (default_value_expr &&
                BiaodashiType::Null == default_value_expr->GetType())
                ARIES_EXCEPTION(ER_INVALID_DEFAULT, column_name.data());
            not_null = true;
        }
    }

    std::string ToString() {
        std::string ret = "";
        ret += this->column_name;
        ret += " ";
        ret += column_type;
        if (this->column_major_len > 0) {
            ret += "(";
            ret += std::to_string(this->column_major_len);
            if (this->column_minor_len >= 0) {
                ret += " , ";
                ret += std::to_string(this->column_minor_len);
            }
            ret += ")";
        }
        ret += " ";
        if (this->not_null) {
            ret += " NOT NULL ";
        }
        if (this->primary_key) {
            ret += " PRIMARY KEY ";
        }
        if (this->foreign_key) {
            ret += " FOREIGN KEY REFERENCES ";
            ret += this->fk_table_name;
            ret += "(";
            ret += this->fk_column_name;
            ret += ")";
        }

        if (default_value && !default_value->empty()) {
            ret += " DEFAULT " + *default_value;
        }

        return ret;
    }
};

typedef std::shared_ptr<ColumnDescription> ColumnDescriptionPointer;


}//ns


#endif
