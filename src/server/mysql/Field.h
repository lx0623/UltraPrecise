//
// Created by tengjp on 19-7-29.
//

#ifndef AIRES_FIELD_H
#define AIRES_FIELD_H


#include <string>

#include "server/mysql/include/m_ctype.h"

using std::string;
class Field {
protected:
    string val;
    uint32	field_length;		// Length of field
public:
    Field(const string& val, uint32 length) {
        field_length = length;
        this->val = val;
    }
    /**
      Writes a copy of the current value in the record buffer, suitable for
      sorting using byte-by-byte comparison. Integers are always in big-endian
      regardless of hardware architecture. At most length bytes are written
      into the buffer.

      @param buff The buffer, assumed to be at least length bytes.

      @param length Number of bytes to write.
    */
    virtual void make_sort_key(uchar *buff, uint length) = 0;

};
class Field_num :public Field {
public:
    bool zerofill,unsigned_flag;	// Purify cannot handle bit fields
    Field_num(const string& val, uint32 len_arg, bool unsigned_arg);
};

class Field_int : public Field_num {
public:
    Field_int(const string& val, uint32 len_arg, bool unsigned_arg);
    static const int PACK_LENGTH= 4;
    void make_sort_key(uchar *buff, uint length);
    uint32 pack_length() const { return PACK_LENGTH; }
};

class Field_float :public Field_num {
public:
    Field_float(const string& val, uint32 len_arg, bool unsigned_arg) :
            Field_num(val, len_arg, unsigned_arg)
    {}
    void make_sort_key(uchar *buff, uint length);
};
class Field_str :public Field {
protected:
    const CHARSET_INFO *field_charset;
public:
    Field_str(const string& val, CHARSET_INFO* charset) :
    Field(val, val.length()){
        field_charset = charset;
    }
    const CHARSET_INFO *charset(void) const { return field_charset; }
    void set_charset(const CHARSET_INFO *charset_arg)
    { field_charset= charset_arg; }
};

class Field_string :public Field_str {
public:
    Field_string(const string& val, CHARSET_INFO* charset) :
            Field_str(val, charset)
    {}
    /* The max. number of characters */
    virtual uint32 char_length() const
    {
        return field_length / charset()->mbmaxlen;
    }

    void make_sort_key(uchar *buff, uint length);
};

#endif //AIRES_FIELD_H
