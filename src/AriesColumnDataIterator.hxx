 #ifndef ARIES_COLUMN_DATAITERATOR_H_
 #define ARIES_COLUMN_DATAITERATOR_H_

 #include "AriesDefinition.h"
 #include "AriesColumnType.h"
 BEGIN_ARIES_ACC_NAMESPACE

    struct AriesColumnDataIterator
    {
        int8_t** m_data;
        size_t m_itemCount;
        aries::AriesColumnType m_valueType;
        int64_t* m_dataBlockSizePrefixSum;
        int32_t m_dataBlockCount;

        int8_t** m_indices;
        size_t m_indiceItemCount;
        aries::AriesColumnType m_indiceValueType;
        int64_t* m_indiceBlockSizePrefixSum;
        int32_t m_indiceBlockCount;

        int8_t* m_nullData;
        size_t m_perItemSize;
        bool m_hasNull;

        ARIES_HOST_DEVICE AriesColumnDataIterator()
        : m_data( nullptr ),
          m_itemCount( 0 ),
          m_dataBlockSizePrefixSum( nullptr ),
          m_dataBlockCount( 0 ),
          m_indices( nullptr ),
          m_indiceItemCount( 0 ),
          m_indiceBlockSizePrefixSum( nullptr ),
          m_indiceBlockCount( 0 ),
          m_nullData( nullptr ),
          m_perItemSize( 0 ),
          m_hasNull( false )
        {
        }

        ARIES_HOST_DEVICE_NO_INLINE int8_t* operator[]( int i ) const;

    };

END_ARIES_ACC_NAMESPACE
#endif /* ARIES_COLUMN_DATAITERATOR_H_ */