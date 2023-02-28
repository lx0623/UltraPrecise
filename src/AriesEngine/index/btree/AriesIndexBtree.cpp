/*
 * AriesIndexBtree.cpp
 *
 *  Created on: Apr 21, 2020
 *      Author: lichi
 */
#include "../AriesIndex.h"
#include "AriesIndexBtree.h"

BEGIN_ARIES_ENGINE_NAMESPACE
using aries::AriesValueType;

    IAriesIndexSPtr AriesIndexCreator::CreateAriesIndex( const std::vector< aries::AriesColumnType >& types, key_hint_t hint )
    {
        IAriesIndexSPtr result;
        assert( types.size() > 1 );
        switch( hint )
        {
            case key_hint_t::most_unique:
            {
                result = std::make_shared< AriesIndex< AriesCompositeKeyType, key_hint_t::most_unique > >( types );
                break;
            }
            case key_hint_t::most_duplicate:
            {
                result = std::make_shared< AriesIndex< AriesCompositeKeyType, key_hint_t::most_duplicate > >( types );
                break;
            }
            default:
                assert( 0 );
                break;
        }
        return result;
    }

    IAriesIndexSPtr AriesIndexCreator::CreateAriesIndex( const aries::AriesColumnType& type, key_hint_t hint )
    {
        IAriesIndexSPtr result;
        switch( hint )
        {
            case key_hint_t::most_unique:
            {
                switch( type.DataType.ValueType )
                {
                    case AriesValueType::INT8:
                        if( !type.HasNull )
                            result = std::make_shared< AriesIndex< int8_t, key_hint_t::most_unique > >( type );
                        else
                            result = std::make_shared< AriesIndex< nullable_type< int8_t >, key_hint_t::most_unique > >( type );
                        break;
                    case AriesValueType::UINT8:
                        if( !type.HasNull )
                            result = std::make_shared< AriesIndex< uint8_t, key_hint_t::most_unique > >( type );
                        else
                            result = std::make_shared< AriesIndex< nullable_type< uint8_t >, key_hint_t::most_unique > >( type );
                        break;
                    case AriesValueType::BOOL:
                        if( !type.HasNull )
                            result = std::make_shared< AriesIndex< bool, key_hint_t::most_unique > >( type );
                        else
                            result = std::make_shared< AriesIndex< nullable_type< bool >, key_hint_t::most_unique > >( type );
                        break;
                    case AriesValueType::CHAR:
                        if( !type.HasNull )
                        {
                            if( type.DataType.Length > 1 )
                                result = std::make_shared< AriesIndex< std::string, key_hint_t::most_unique > >( type );
                            else
                                result = std::make_shared< AriesIndex< char, key_hint_t::most_unique > >( type );
                        }
                        else
                        {
                            if( type.DataType.Length > 1 )
                                result = std::make_shared< AriesIndex< unpacked_nullable_type< std::string >, key_hint_t::most_unique > >( type );
                            else
                                result = std::make_shared< AriesIndex< nullable_type< char >, key_hint_t::most_unique > >( type );
                        }
                        break;
                    case AriesValueType::COMPACT_DECIMAL:
                        //lichi: We should always use Decimal for index not compact decimal
                        if( !type.HasNull )
                            result = std::make_shared< AriesIndex< aries_acc::Decimal, key_hint_t::most_unique > >( type );
                        else
                            result = std::make_shared< AriesIndex< nullable_type< aries_acc::Decimal >, key_hint_t::most_unique > >( type );
                        break;
                    case AriesValueType::INT16:
                        if( !type.HasNull )
                            result = std::make_shared< AriesIndex< int16_t, key_hint_t::most_unique > >( type );
                        else
                            result = std::make_shared< AriesIndex< nullable_type< int16_t >, key_hint_t::most_unique > >( type );
                        break;
                    case AriesValueType::UINT16:
                        if( !type.HasNull )
                            result = std::make_shared< AriesIndex< uint16_t, key_hint_t::most_unique > >( type );
                        else
                            result = std::make_shared< AriesIndex< nullable_type< uint16_t >, key_hint_t::most_unique > >( type );
                        break;
                    case AriesValueType::INT32:
                        if( !type.HasNull )
                            result = std::make_shared< AriesIndex< int32_t, key_hint_t::most_unique > >( type );
                        else
                            result = std::make_shared< AriesIndex< nullable_type< int32_t >, key_hint_t::most_unique > >( type );
                        break;
                    case AriesValueType::UINT32:
                        if( !type.HasNull )
                            result = std::make_shared< AriesIndex< uint32_t, key_hint_t::most_unique > >( type );
                        else
                            result = std::make_shared< AriesIndex< nullable_type< uint32_t >, key_hint_t::most_unique > >( type );
                        break;
                    case AriesValueType::INT64:
                        if( !type.HasNull )
                            result = std::make_shared< AriesIndex< int64_t, key_hint_t::most_unique > >( type );
                        else
                            result = std::make_shared< AriesIndex< nullable_type< int64_t >, key_hint_t::most_unique > >( type );
                        break;
                    case AriesValueType::UINT64:
                        if( !type.HasNull )
                            result = std::make_shared< AriesIndex< uint64_t, key_hint_t::most_unique > >( type );
                        break;
                    case AriesValueType::FLOAT:
                        if( !type.HasNull )
                            result = std::make_shared< AriesIndex< float, key_hint_t::most_unique > >( type );
                        else
                            result = std::make_shared< AriesIndex< nullable_type< float >, key_hint_t::most_unique > >( type );
                        break;
                    case AriesValueType::DOUBLE:
                        if( !type.HasNull )
                            result = std::make_shared< AriesIndex< double, key_hint_t::most_unique > >( type );
                        else
                            result = std::make_shared< AriesIndex< nullable_type< double >, key_hint_t::most_unique > >( type );
                        break;
                    case AriesValueType::DECIMAL:
                        if( !type.HasNull )
                            result = std::make_shared< AriesIndex< aries_acc::Decimal, key_hint_t::most_unique > >( type );
                        else
                            result = std::make_shared< AriesIndex< nullable_type< aries_acc::Decimal >, key_hint_t::most_unique > >( type );
                        break;
                    case AriesValueType::DATE:
                        if( !type.HasNull )
                            result = std::make_shared< AriesIndex< aries_acc::AriesDate, key_hint_t::most_unique > >( type );
                        else
                            result = std::make_shared< AriesIndex< nullable_type< aries_acc::AriesDate >, key_hint_t::most_unique > >( type );
                        break;
                    case AriesValueType::DATETIME:
                        if( !type.HasNull )
                            result = std::make_shared< AriesIndex< aries_acc::AriesDatetime, key_hint_t::most_unique > >( type );
                        else
                            result = std::make_shared< AriesIndex< nullable_type< aries_acc::AriesDatetime >, key_hint_t::most_unique > >( type );
                        break;
                    case AriesValueType::TIMESTAMP:
                        if( !type.HasNull )
                            result = std::make_shared< AriesIndex< aries_acc::AriesTimestamp, key_hint_t::most_unique > >( type );
                        else
                            result = std::make_shared< AriesIndex< nullable_type< aries_acc::AriesTimestamp >, key_hint_t::most_unique > >( type );
                        break;
                    case AriesValueType::TIME:
                        if( !type.HasNull )
                            result = std::make_shared< AriesIndex< aries_acc::AriesTime, key_hint_t::most_unique > >( type );
                        else
                            result = std::make_shared< AriesIndex< nullable_type< aries_acc::AriesTime >, key_hint_t::most_unique > >( type );
                        break;
                    case AriesValueType::YEAR:
                        if( !type.HasNull )
                            result = std::make_shared< AriesIndex< aries_acc::AriesYear, key_hint_t::most_unique > >( type );
                        else
                            result = std::make_shared< AriesIndex< nullable_type< aries_acc::AriesYear >, key_hint_t::most_unique > >( type );
                        break;
                    default:
                        assert( 0 );
                        break;
                }
                break;
            }
            case key_hint_t::most_duplicate:
            {
                switch( type.DataType.ValueType )
                {
                    case AriesValueType::INT8:
                        if( !type.HasNull )
                            result = std::make_shared< AriesIndex< int8_t, key_hint_t::most_duplicate > >( type );
                        else
                            result = std::make_shared< AriesIndex< nullable_type< int8_t >, key_hint_t::most_duplicate > >( type );
                        break;
                    case AriesValueType::UINT8:
                        if( !type.HasNull )
                            result = std::make_shared< AriesIndex< uint8_t, key_hint_t::most_duplicate > >( type );
                        else
                            result = std::make_shared< AriesIndex< nullable_type< uint8_t >, key_hint_t::most_duplicate > >( type );
                        break;
                    case AriesValueType::BOOL:
                        if( !type.HasNull )
                            result = std::make_shared< AriesIndex< bool, key_hint_t::most_duplicate > >( type );
                        else
                            result = std::make_shared< AriesIndex< nullable_type< bool >, key_hint_t::most_duplicate > >( type );
                        break;
                    case AriesValueType::CHAR:
                        if( !type.HasNull )
                        {
                            if( type.DataType.Length > 1 )
                                result = std::make_shared< AriesIndex< std::string, key_hint_t::most_duplicate > >( type );
                            else
                                result = std::make_shared< AriesIndex< char, key_hint_t::most_duplicate > >( type );
                        }
                        else
                        {
                            if( type.DataType.Length > 1 )
                                result = std::make_shared< AriesIndex< unpacked_nullable_type< std::string >, key_hint_t::most_duplicate > >( type );
                            else
                                result = std::make_shared< AriesIndex< nullable_type< char >, key_hint_t::most_duplicate > >( type );
                        }
                        break;
                    case AriesValueType::COMPACT_DECIMAL:
                        //lichi: We should always use Decimal for index not compact decimal
                        if( !type.HasNull )
                            result = std::make_shared< AriesIndex< aries_acc::Decimal, key_hint_t::most_duplicate > >( type );
                        else
                            result = std::make_shared< AriesIndex< nullable_type< aries_acc::Decimal >, key_hint_t::most_duplicate > >( type );
                        break;
                    case AriesValueType::INT16:
                        if( !type.HasNull )
                            result = std::make_shared< AriesIndex< int16_t, key_hint_t::most_duplicate > >( type );
                        else
                            result = std::make_shared< AriesIndex< nullable_type< int16_t >, key_hint_t::most_duplicate > >( type );
                        break;
                    case AriesValueType::UINT16:
                        if( !type.HasNull )
                            result = std::make_shared< AriesIndex< uint16_t, key_hint_t::most_duplicate > >( type );
                        else
                            result = std::make_shared< AriesIndex< nullable_type< uint16_t >, key_hint_t::most_duplicate > >( type );
                        break;
                    case AriesValueType::INT32:
                        if( !type.HasNull )
                            result = std::make_shared< AriesIndex< int32_t, key_hint_t::most_duplicate > >( type );
                        else
                            result = std::make_shared< AriesIndex< nullable_type< int32_t >, key_hint_t::most_duplicate > >( type );
                        break;
                    case AriesValueType::UINT32:
                        if( !type.HasNull )
                            result = std::make_shared< AriesIndex< uint32_t, key_hint_t::most_duplicate > >( type );
                        else
                            result = std::make_shared< AriesIndex< nullable_type< uint32_t >, key_hint_t::most_duplicate > >( type );
                        break;
                    case AriesValueType::INT64:
                        if( !type.HasNull )
                            result = std::make_shared< AriesIndex< int64_t, key_hint_t::most_duplicate > >( type );
                        else
                            result = std::make_shared< AriesIndex< nullable_type< int64_t >, key_hint_t::most_duplicate > >( type );
                        break;
                    case AriesValueType::UINT64:
                        if( !type.HasNull )
                            result = std::make_shared< AriesIndex< uint64_t, key_hint_t::most_duplicate > >( type );
                        break;
                    case AriesValueType::FLOAT:
                        if( !type.HasNull )
                            result = std::make_shared< AriesIndex< float, key_hint_t::most_duplicate > >( type );
                        else
                            result = std::make_shared< AriesIndex< nullable_type< float >, key_hint_t::most_duplicate > >( type );
                        break;
                    case AriesValueType::DOUBLE:
                        if( !type.HasNull )
                            result = std::make_shared< AriesIndex< double, key_hint_t::most_duplicate > >( type );
                        else
                            result = std::make_shared< AriesIndex< nullable_type< double >, key_hint_t::most_duplicate > >( type );
                        break;
                    case AriesValueType::DECIMAL:
                        if( !type.HasNull )
                            result = std::make_shared< AriesIndex< aries_acc::Decimal, key_hint_t::most_duplicate > >( type );
                        else
                            result = std::make_shared< AriesIndex< nullable_type< aries_acc::Decimal >, key_hint_t::most_duplicate > >( type );
                        break;
                    case AriesValueType::DATE:
                        if( !type.HasNull )
                            result = std::make_shared< AriesIndex< aries_acc::AriesDate, key_hint_t::most_duplicate > >( type );
                        else
                            result = std::make_shared< AriesIndex< nullable_type< aries_acc::AriesDate >, key_hint_t::most_duplicate > >( type );
                        break;
                    case AriesValueType::DATETIME:
                        if( !type.HasNull )
                            result = std::make_shared< AriesIndex< aries_acc::AriesDatetime, key_hint_t::most_duplicate > >( type );
                        else
                            result = std::make_shared< AriesIndex< nullable_type< aries_acc::AriesDatetime >, key_hint_t::most_duplicate > >( type );
                        break;
                    case AriesValueType::TIMESTAMP:
                        if( !type.HasNull )
                            result = std::make_shared< AriesIndex< aries_acc::AriesTimestamp, key_hint_t::most_duplicate > >( type );
                        else
                            result = std::make_shared< AriesIndex< nullable_type< aries_acc::AriesTimestamp >, key_hint_t::most_duplicate > >( type );
                        break;
                    case AriesValueType::TIME:
                        if( !type.HasNull )
                            result = std::make_shared< AriesIndex< aries_acc::AriesTime, key_hint_t::most_duplicate > >( type );
                        else
                            result = std::make_shared< AriesIndex< nullable_type< aries_acc::AriesTime >, key_hint_t::most_duplicate > >( type );
                        break;
                    case AriesValueType::YEAR:
                        if( !type.HasNull )
                            result = std::make_shared< AriesIndex< aries_acc::AriesYear, key_hint_t::most_duplicate > >( type );
                        else
                            result = std::make_shared< AriesIndex< nullable_type< aries_acc::AriesYear >, key_hint_t::most_duplicate > >( type );
                        break;
                    default:
                        assert( 0 );
                        break;
                }
                break;
            }
            default:
                assert( 0 );
                break;
        }
        return result;
    }

    // IAriesIndexSPtr AriesIndexCreator::CreateAriesIndexKeyPosition( const aries_acc::AriesDictKeyComparator& comp, key_hint_t hint )
    // {
    //     IAriesIndexSPtr result;
    //     switch( hint )
    //     {
    //         case key_hint_t::most_unique:
    //         {
    //             result = std::make_shared< AriesIndex< KeyPosition, key_hint_t::most_unique > >( comp );
    //             break;
    //         }
    //         case key_hint_t::most_duplicate:
    //         {
    //             result = std::make_shared< AriesIndex< KeyPosition, key_hint_t::most_duplicate > >( comp );
    //             break;
    //         }
    //         default:
    //             assert( 0 );
    //             break;
    //     }
    //     return result;
    // }

END_ARIES_ENGINE_NAMESPACE
