/*
 * cpptraits.hxx
 *
 *  Created on: Jul 20, 2019
 *      Author: lichi
 */

#ifndef CPPTRAITS_HXX_
#define CPPTRAITS_HXX_
#include "AriesDefinition.h"

BEGIN_ARIES_ACC_NAMESPACE

/// integral_constant
template< typename _Tp, _Tp __v >
struct integral_constant
{
    static constexpr _Tp value = __v;
    typedef _Tp value_type;
    typedef integral_constant< _Tp, __v > type;
    ARIES_HOST_DEVICE constexpr operator value_type() const noexcept
    {
        return value;
    }
#if __cplusplus > 201103L

#define __cpp_lib_integral_constant_callable 201304

    ARIES_HOST_DEVICE constexpr value_type operator()() const noexcept
    {   return value;}
#endif
};

template< typename _Tp, _Tp __v >
constexpr _Tp integral_constant< _Tp, __v >::value;

/// The type used as a compile-time boolean with true value.
typedef integral_constant< bool, true > true_type;

/// The type used as a compile-time boolean with false value.
typedef integral_constant< bool, false > false_type;

template< bool __v >
using __bool_constant = integral_constant<bool, __v>;

#if __cplusplus > 201402L
# define __cpp_lib_bool_constant 201505
template<bool __v>
using bool_constant = integral_constant<bool, __v>;
#endif

// Meta programming helper types.

template< bool, typename, typename >
struct conditional;
template< bool _Cond, typename _Iftrue, typename _Iffalse >
struct conditional
{
    typedef _Iftrue type;
};

// Partial specialization for false.
template< typename _Iftrue, typename _Iffalse >
struct conditional< false, _Iftrue, _Iffalse >
{
    typedef _Iffalse type;
};

template< typename ...>
struct __or_;

template< >
struct __or_< > : public false_type
{
};

template< typename _B1 >
struct __or_< _B1 > : public _B1
{
};

template< typename _B1, typename _B2 >
struct __or_< _B1, _B2 > : public conditional< _B1::value, _B1, _B2 >::type
{
};

template< typename _Tp >
struct remove_const
{
    typedef _Tp type;
};

template< typename _Tp >
struct remove_const< _Tp const >
{
    typedef _Tp type;
};

/// remove_volatile
template< typename _Tp >
struct remove_volatile
{
    typedef _Tp type;
};

template< typename _Tp >
struct remove_volatile< _Tp volatile >
{
    typedef _Tp type;
};

/// remove_cv
template< typename _Tp >
struct remove_cv
{
    typedef typename remove_const< typename remove_volatile< _Tp >::type >::type type;
};

template< typename >
struct __is_integral_helper: public false_type
{
};

template< >
struct __is_integral_helper< bool > : public true_type
{
};

template< >
struct __is_integral_helper< char > : public true_type
{
};

template< >
struct __is_integral_helper< signed char > : public true_type
{
};

template< >
struct __is_integral_helper< unsigned char > : public true_type
{
};

#ifdef _GLIBCXX_USE_WCHAR_T
template<>
struct __is_integral_helper<wchar_t>
: public true_type
{};
#endif

template< >
struct __is_integral_helper< char16_t > : public true_type
{
};

template< >
struct __is_integral_helper< char32_t > : public true_type
{
};

template< >
struct __is_integral_helper< short > : public true_type
{
};

template< >
struct __is_integral_helper< unsigned short > : public true_type
{
};

template< >
struct __is_integral_helper< int > : public true_type
{
};

template< >
struct __is_integral_helper< unsigned int > : public true_type
{
};

template< >
struct __is_integral_helper< long > : public true_type
{
};

template< >
struct __is_integral_helper< unsigned long > : public true_type
{
};

template< >
struct __is_integral_helper< long long > : public true_type
{
};

template< >
struct __is_integral_helper< unsigned long long > : public true_type
{
};
template< typename _Tp >
struct is_integral: public __is_integral_helper< typename remove_cv< _Tp >::type >::type
{
};

template< typename >
struct __is_floating_point_helper: public false_type
{
};

template< >
struct __is_floating_point_helper< float > : public true_type
{
};

template< >
struct __is_floating_point_helper< double > : public true_type
{
};

template< >
struct __is_floating_point_helper< long double > : public true_type
{
};

#if !defined(__STRICT_ANSI__) && defined(_GLIBCXX_USE_FLOAT128) && !defined(__CUDACC__)
template<>
struct __is_floating_point_helper<__float128>
: public true_type
{};
#endif

/// is_floating_point
template< typename _Tp >
struct is_floating_point: public __is_floating_point_helper< typename remove_cv< _Tp >::type >::type
{
};

template< typename _Tp >
struct is_arithmetic: public __or_< is_integral< _Tp >, is_floating_point< _Tp > >::type
{
};

template< bool, typename _Tp = void >
struct enable_if
{
};

// Partial specialization for true.
template< typename _Tp >
struct enable_if< true, _Tp >
{
    typedef _Tp type;
};

/// is_same
template< typename, typename >
struct is_same: public false_type
{
};

template< typename _Tp >
struct is_same< _Tp, _Tp > : public true_type
{
};

END_ARIES_ACC_NAMESPACE

#endif /* CPPTRAITS_HXX_ */
