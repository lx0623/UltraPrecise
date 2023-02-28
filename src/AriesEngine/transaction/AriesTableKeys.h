/*！
 * @file AriesTableKeys.h
 * 
 * 
 */

#ifndef ARIESTABLEKEYS_H_
#define ARIESTABLEKEYS_H_
#include <mutex>
#include <unordered_map>
#include "../AriesDataDef.h"
#include "AriesSimpleItemContainer.h"

BEGIN_ARIES_ENGINE_NAMESPACE
 /*!
    @brief 处理主键冲突检测的辅助数据结构

     主键来自initial table 和 delta table，分别建立两个数据结构
    - 排序数组: 启动数据库时建立，对initial table中的key排序，并收集每个key对应的rowPos集合；
    - map(key->rowPos): 在运行期建立，如果插入的tuple的key不在排序数组中，则插入到map中。
   
    @note
    - 因为mvcc，一个key可能对应多个版本，所以无论是排序数组还是map，每个key都可能对应对多rowPos
    - 删除tuple时，tuple对应的key不会从排序数组和map中删除 （tuple 所在slot的header会修改xmax）,因此通过key在TableKeys查到的slot的key值可能已经改变
    - 如果key在排序数组中出现，那么不会在map中出现
    - 检查主键冲突时，AriesTableKeys 通过排序数组和map返回一组可能的rowPos，调用者需要通过rowPos得到对应slot，通过mvcc算法判断是否有主键冲突
    - 对rowPos的变长数组的操作和string类似，因此借用string来表示动态数组
    - 无论是多字段还是单个字段，primary key 都用string表示
    - 因为前期还兼顾处理外键，所以部分函数的参数有冗余，需要重构

 */
    using AriesRowPosContainer = aries::AriesSimpleItemContainer< RowPos >;
    class AriesTableKeys;
    using AriesTableKeysSPtr = std::shared_ptr< AriesTableKeys >;
    class AriesTableKeys
    {
    public:
        AriesTableKeys();
        ~AriesTableKeys();

        bool Merge( AriesTableKeysSPtr incr );

        /*!
        @brief 对initial table中的主键字段排序
        @param[in] columns - 组成 primary key 的列
        @param[in] checkDuplicate 主键:true,外键:false 
        @return
            @retval true 初始数据没有重复值
            @retval false 当主键重复时，返回false

        @note 处理外键是历史遗留，因此refactor后应该去掉 checkDuplicate 
        */
        bool Build( const std::vector< aries_engine::AriesColumnSPtr >& columns, bool checkDuplicate = true, size_t reservedDeltaCount = 0 );

        /*!
        @brief 将 <key,rowPos>加入TableKeys
        @param[in] key 主键字段二进制拼接后用string保存
        @param[in] rowPos 将要插入的tuple所在的slot
        @param existedLocationCount tableKeys中key对应的slot的数量的预期值
        @return 
        @retval false 插入失败
        @retval true 插入成功
        @note
        分两阶段插入key。
        - 首先通过FindKey得到key对应的slot以及slot数量`existedLocationCount`，检查slot的key字段和mvcc信息。检查通过后写入slot。
        - 写入完成后更新TableKeys。再次查询key对应的slot，如果slot数量有变化，说明已经有trx插入了这个key，InsertKey返回false
        */
        bool InsertKey( const string& key, RowPos rowPos, bool checkDuplicate = true, size_t existedLocationCount = 0 );

        /*!
        @brief 查找key值的rowPos信息．没有相同key值的tuple时，返回( false, nullptr )
        @return 
            @retval (false, nullptr) 没有找到key对应的rowPos
            @retval (true, [rowpos])          
        */
        pair< bool, AriesRowPosContainer > FindKey( const string& key );

    private:
        /*!
        @brief 在排序数组和map中查找key
        @return 
            @retval (false, nullptr) 没有找到key对应的rowPos
            @retval (true, [rowpos]) 
        @note 在排序数组或map中找到的slot还需要经过mvcc才能决定是否有主键冲突
        */
        pair< bool, AriesRowPosContainer* > FindKeyInternal( const string& key );
        /// @brief 在排序数组中二分查找key
        int BinaryFind( const char* keys, size_t len, int count, const char* key );
        /// @brief 在map中查找key
        bool IsKeyInDeltaTable(const string &key);

    private:
        string m_initialKeys; //!<initial table中主键排序的结果.排序数组被看做是一个string
        size_t m_initialKeyCount; //!< inital table中tuple数量
        size_t m_perKeySize; //!< 主键字段的size，如果是联合主键，则将所有主键字段size相加
        vector< AriesRowPosContainer > m_tupleLocations; //vector中元素和initial table中排序的key一一对应，每个元素表示key对应的rowPos集合
        unordered_map< string, AriesRowPosContainer > m_deltaKeys;//!< 运行期增加的key到它们所在slot集合的映射
        mutex m_mutex;
    };

END_ARIES_ENGINE_NAMESPACE
/* namespace aries_engine */

#endif /* ARIESTABLEKEYS_H_ */
