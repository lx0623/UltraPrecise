#include <memory>
#include <vector>
using namespace std;

size_t getConcurrency( size_t totalJobCnt,
                       vector<size_t>& threadsJobCnt,
                       vector<size_t>& threadsJobStartIdx );