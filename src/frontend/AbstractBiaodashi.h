#ifndef ARIES_ABSTRACT_BIAODASHI
#define ARIES_ABSTRACT_BIAODASHI

#include <string>
#include <memory>

namespace aries {

/***************************************************************************
 *
 *                          Everything is a Biaodashi!
 *
 **************************************************************************/



/*The base*/
class AbstractBiaodashi {
public:
    virtual ~AbstractBiaodashi() = default;

    virtual std::string ToString() = 0;

    virtual void AddChild(std::shared_ptr<AbstractBiaodashi> arg_child) = 0;
    virtual std::string GetName() {
        return name;
    }

    void SetName(const std::string& name_arg) {
        name = name_arg;
    }

    void SetParent( AbstractBiaodashi* p )
    {
        parent = p;
    }
    AbstractBiaodashi *GetParent() const
    {
        return parent;
    }
protected:
    std::string name;
    AbstractBiaodashi* parent = nullptr;

};


typedef std::shared_ptr<AbstractBiaodashi> BiaodashiPointer;

}//namespace


#endif
