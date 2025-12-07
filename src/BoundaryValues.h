#pragma once
#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <cmath>

namespace NavierStokes {
using namespace dealii;

template <int dim>
class BoundaryValues : public Function<dim> {
public:
    BoundaryValues();
    virtual double value(const Point<dim> &p, const unsigned int component) const override;
};
} // namespace NavierStokes