%module  instant_module_315ebc21e0fe220e611965734859ae349eef07f9
//%module (directors="1") instant_module_315ebc21e0fe220e611965734859ae349eef07f9

//%feature("director");

%{
#include <iostream>
 
#include <boost/shared_ptr.hpp> 
 
#include "ffc_form_53dbf55d24d551a08237d0fe534a717af84d3787.h" 

%}

//%feature("autodoc", "1");


%init%{

%}



//Uncomment these to produce code for std::tr1::shared_ptr
//#define SWIG_SHARED_PTR_NAMESPACE std
//#define SWIG_SHARED_PTR_SUBNAMESPACE tr1
%include <boost_shared_ptr.i>

// Declare which classes should be stored using shared_ptr
%shared_ptr(ufc::cell_integral)
%shared_ptr(ufc::dofmap)
%shared_ptr(ufc::finite_element)
%shared_ptr(ufc::function)
%shared_ptr(ufc::form)
%shared_ptr(ufc::exterior_facet_integral)
%shared_ptr(ufc::interior_facet_integral)
%shared_ptr(ufc::point_integral)

// Import types from ufc
%import(module="ufc") "ufc.h"

// Swig shared_ptr macro declarations
%shared_ptr(ffc_form_53dbf55d24d551a08237d0fe534a717af84d3787_finite_element_0)
%shared_ptr(ffc_form_53dbf55d24d551a08237d0fe534a717af84d3787_dofmap_0)
%shared_ptr(ffc_form_53dbf55d24d551a08237d0fe534a717af84d3787_cell_integral_0_otherwise)
%shared_ptr(ffc_form_53dbf55d24d551a08237d0fe534a717af84d3787_form_0)

%include "ffc_form_53dbf55d24d551a08237d0fe534a717af84d3787.h"
//
;

