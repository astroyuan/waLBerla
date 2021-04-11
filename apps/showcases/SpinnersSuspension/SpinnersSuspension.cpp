#include "core/all.h"

#include "blockforest/Initialization.h"
#include "blockforest/communication/all.h"

#include "pe/basic.h"

#include <tuple>

namespace spinners_suspension
{
using namespace walberla;
using walberla::uint_t;

// body types
using BodyTypeTuple = std::tuple< pe::Plane, pe::Sphere>;

/////////////////////
// parameters
/////////////////////
struct Setup
{
    // domain parameters
    Vector3< uint_t > numBlocks; // number of blocks in x,y,z direction
    Vector3< uint_t > domainSize; // domian size in x,y,z direction
    Vector3< bool > isPeriodic; // whether periodic in x,y,z direction

    // material properties
    real_t particle_density; // density of particles
    real_t fluid_density; // density of fluid
    real_t density_ratio;

    real_t particle_diameter_1; // particle diamter first type
    real_t particle_diameter_2; // particle diamter second type
    real_t particle_diameter_avg; // average particle diameter

    void printSetup()
    {
        WALBERLA_LOG_DEVEL_ON_ROOT(numBlocks);
        WALBERLA_LOG_DEVEL_ON_ROOT(domainSize);
        WALBERLA_LOG_DEVEL_ON_ROOT(isPeriodic);

        WALBERLA_LOG_DEVEL_ON_ROOT(particle_density);
        WALBERLA_LOG_DEVEL_ON_ROOT(fluid_density);
        WALBERLA_LOG_DEVEL_ON_ROOT(density_ratio);

        WALBERLA_LOG_DEVEL_ON_ROOT(particle_diameter_1);
        WALBERLA_LOG_DEVEL_ON_ROOT(particle_diameter_2);
        WALBERLA_LOG_DEVEL_ON_ROOT(particle_diameter_avg);
    }

    void sanity_check()
    {
        if ( isPeriodic[0] == true && numBlocks[0] < uint_c(3) )
            WALBERLA_ABORT("The number of blocks in any periodic direction must not be smaller than 3.");
        if ( isPeriodic[1] == true && numBlocks[1] < uint_c(3) )
            WALBERLA_ABORT("The number of blocks in any periodic direction must not be smaller than 3.");
        if ( isPeriodic[2] == true && numBlocks[2] < uint_c(3) )
            WALBERLA_ABORT("The number of blocks in any periodic direction must not be smaller than 3.");

        WALBERLA_CHECK(particle_diameter_1 > 0 && particle_diameter_2 > 0, "particle diamter should be positive.");
    }
};


//*******************************************************************************************************************
/*!\brief Simulation of a collection rotating colloids with the discrete particle method.
 *
 * The simulation features a fluidized bed with spherical particles inside a rectangular column.
 * The domain size is [32 x 16 x 256] * d_avg
 *
 */
//*******************************************************************************************************************
int main(int argc, char** argv)
{
    Environment env(argc, argv);

    if (argc < 2) { WALBERLA_ABORT("Please specify a parameter file as input argument."); }

    Setup setup;


    //////////////////////////////////
    // Parse simulation parameters
    //////////////////////////////////

    auto domainParameters = env.config()->getOneBlock("DomainParameters");
    setup.numBlocks = domainParameters.getParameter< Vector3< uint_t > >("numBlocks");
    setup.domainSize = domainParameters.getParameter< Vector3< uint_t > >("domainSize");
    setup.isPeriodic = domainParameters.getParameter < Vector3<  bool > >("isPeriodic");

    auto simulationParameters = env.config()->getOneBlock("SimulationParameters");
    //pending

    auto materialProperties = env.config()->getOneBlock("MaterialProperties");
    setup.particle_density = materialProperties.getParameter< real_t >("particle_density");
    setup.fluid_density = materialProperties.getParameter< real_t >("fluid_density");
    setup.density_ratio = setup.particle_density / setup.fluid_density;

    setup.particle_diameter_1 = materialProperties.getParameter< real_t >("particle_diameter_1");
    setup.particle_diameter_2 = materialProperties.getParameter< real_t >("particle_diameter_2");
    setup.particle_diameter_avg = (setup.particle_diameter_1 + setup.particle_diameter_2) / real_t(2.0);

    auto outputParameters = env.config()->getOneBlock("OutputParameters");
    //pending

    // check sanity of input parameters
    //setup.sanity_check();

    std::cout << std::boolalpha;
    std::cout << setup.isPeriodic[0] << std::endl;
    
    setup.printSetup();

    ////////////////////////////
    // simulation setup
    ////////////////////////////

    return EXIT_SUCCESS;
}

} //namespace spinners_suspension_dpm

int main( int argc, char **argv){
    return spinners_suspension::main(argc, argv);
}