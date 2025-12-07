#include "BoundaryValues.h"
#include <cmath>

namespace NavierStokes {
using namespace dealii;

template <int dim>
BoundaryValues<dim>::BoundaryValues() : Function<dim>(dim + 1) {}

template <int dim>
double BoundaryValues<dim>::value(const Point<dim> &p, const unsigned int component) const {
    Assert(component < this->n_components,
           ExcIndexRange(component, 0, this->n_components));
    if (component == 0 && std::abs(p[dim - 1] - 1.0) < 1e-10)
        return 1.0;
    return 0;
}

// Explicit instantiation for dim=2
template class BoundaryValues<2>;
} // namespace NavierStokes
