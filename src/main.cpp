#include "StationaryNavierStokes.hpp"
#include <iostream>

int main() {
    using namespace NavierStokes;
    try {
        StationaryNavierStokes<2> flow(/* degree = */ 1);
        flow.run(4);
    } catch (std::exception &exc) {
        std::cerr << std::endl << std::endl
                  << "----------------------------------------------------" << std::endl;
        std::cerr << "Exception on processing: " << std::endl
                  << exc.what() << std::endl
                  << "Aborting!" << std::endl
                  << "----------------------------------------------------" << std::endl;
        return 1;
    } catch (...) {
        std::cerr << std::endl << std::endl
                  << "----------------------------------------------------" << std::endl;
        std::cerr << "Unknown exception!" << std::endl
                  << "Aborting!" << std::endl
                  << "----------------------------------------------------" << std::endl;
        return 1;
    }
    return 0;
}
