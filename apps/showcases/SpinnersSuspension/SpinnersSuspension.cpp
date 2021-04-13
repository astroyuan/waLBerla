#include "core/all.h"

#include "blockforest/Initialization.h"
#include "blockforest/communication/all.h"

#include "core/logging/all.h"

#include "pe/basic.h"
#include "pe/Types.h"
#include "pe/synchronization/SyncShadowOwners.h"
#include "pe/utility/DestroyBody.h"
#include "pe/vtk/SphereVtkOutput.h"

#include "vtk/VTKOutput.h"

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

    // simulation parameters
    uint_t timesteps; // simulation time steps

    // material properties
    real_t particle_density; // density of particles
    real_t fluid_density; // density of fluid
    real_t density_ratio;

    real_t particle_diameter_1; // particle diameter first type
    real_t particle_diameter_2; // particle diameter second type
    real_t particle_diameter_max; // maxium possible particle diameters
    real_t particle_diameter_avg; // average particle diameter
    real_t particle_volume_avg; // average particle volume
    real_t particle_mass_avg; // average particle mass

    real_t restitutionCoeff; // Coefficient of restitution
    real_t frictionSCoeff; // Coefficient of static friction
    real_t frictionDCoeff; // Coefficient of dynamic friction
    real_t poisson; // Poisson's ratio
    real_t young; // Young's modulus
    real_t contactT; // contact time
    real_t stiffnessCoeff; // contact stiffness
    real_t dampingNCoeff; // damping coefficient in the normal direction
    real_t dampingTCoeff; // damping coefficient in the tangential direction

    // output parameters
    std::string vtkBaseFolder;
    uint_t vtkWriteFrequency;
    uint_t logInfoFrequency;

    void printSetup()
    {
        WALBERLA_LOG_DEVEL_VAR_ON_ROOT(numBlocks);
        WALBERLA_LOG_DEVEL_VAR_ON_ROOT(domainSize);
        WALBERLA_LOG_DEVEL_VAR_ON_ROOT(isPeriodic);

        WALBERLA_LOG_DEVEL_VAR_ON_ROOT(particle_density);
        WALBERLA_LOG_DEVEL_VAR_ON_ROOT(fluid_density);
        WALBERLA_LOG_DEVEL_VAR_ON_ROOT(density_ratio);

        WALBERLA_LOG_DEVEL_VAR_ON_ROOT(particle_diameter_1);
        WALBERLA_LOG_DEVEL_VAR_ON_ROOT(particle_diameter_2);
        WALBERLA_LOG_DEVEL_VAR_ON_ROOT(particle_diameter_avg);
        WALBERLA_LOG_DEVEL_VAR_ON_ROOT(particle_volume_avg);
        WALBERLA_LOG_DEVEL_VAR_ON_ROOT(particle_mass_avg);

        WALBERLA_LOG_DEVEL_VAR_ON_ROOT(vtkBaseFolder);
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

uint_t createMonolayerSpheresSC(BlockForest & forest, pe::BodyStorage & globalBodyStorage, const BlockDataID & bodyStorageID,
                                const Setup & setup, const AABB & domainSimulation, pe::MaterialID & material)
{
    /*
    Create a monolayer of spherical particles on a simple cubic lattice
    */
    real_t thickness = 1.0 * setup.particle_diameter_avg;
    real_t spacing = 1.0 * setup.particle_diameter_avg;
    real_t noise_level = 0.1;

    const auto domainGeneration = AABB(real_c(0.0), real_c(0.0), real_c(0.0),
    domainSimulation.xMax(), domainSimulation.yMax(), thickness);

    const Vector3< real_t > pointRef(real_c(0.5)*setup.particle_diameter_max, real_c(0.5)*setup.particle_diameter_max, real_c(0.5)*setup.particle_diameter_max);

    uint_t numParticles = 0;

    for (auto it = grid_generator::SCIterator(domainGeneration, pointRef, spacing); it != grid_generator::SCIterator(); ++it)
    {
        real_t radius = setup.particle_diameter_avg / real_c(2.0);

        // random perturbation
        Vector3< real_t > rndDisp(noise_level * math::realRandom< real_t >(-spacing, spacing),
                                  noise_level * math::realRandom< real_t >(-spacing, spacing),
                                  noise_level * math::realRandom< real_t >(-spacing, spacing));
        // create spheres
        pe::SphereID sp = pe::createSphere(globalBodyStorage, forest, bodyStorageID, 0, *it + rndDisp, radius, material);

        if (sp != nullptr) ++numParticles;
    }

    return numParticles;
}

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
    setup.timesteps = simulationParameters.getParameter< uint_t >("timesteps");

    auto materialProperties = env.config()->getOneBlock("MaterialProperties");
    setup.particle_density = materialProperties.getParameter< real_t >("particle_density");
    setup.fluid_density = materialProperties.getParameter< real_t >("fluid_density");
    setup.density_ratio = setup.particle_density / setup.fluid_density;

    setup.particle_diameter_1 = materialProperties.getParameter< real_t >("particle_diameter_1");
    setup.particle_diameter_2 = materialProperties.getParameter< real_t >("particle_diameter_2");
    setup.particle_diameter_max = std::max(setup.particle_diameter_1, setup.particle_diameter_2);
    setup.particle_diameter_avg = (setup.particle_diameter_1 + setup.particle_diameter_2) / real_t(2.0);
    setup.particle_volume_avg = setup.particle_diameter_avg * setup.particle_diameter_avg * setup.particle_diameter_avg * math::pi / real_t(6.0);
    setup.particle_mass_avg = setup.density_ratio * setup.particle_volume_avg;

    // collision related material properties
    setup.restitutionCoeff = materialProperties.getParameter< real_t >("restitutionCoeff");
    setup.frictionSCoeff = materialProperties.getParameter< real_t >("frictionSCoeff");
    setup.frictionDCoeff = materialProperties.getParameter< real_t >("frictionDCoeff");
    setup.poisson = materialProperties.getParameter< real_t >("poisson");
    setup.young = materialProperties.getParameter< real_t >("young");
    setup.contactT = materialProperties.getParameter< real_t >("contactT");

    real_t mij = setup.particle_mass_avg * setup.particle_mass_avg / (setup.particle_mass_avg + setup.particle_mass_avg);

    setup.stiffnessCoeff = math::pi * math::pi * mij / (setup.contactT * setup.contactT * (real_t(1.0) - 
    std::log(setup.restitutionCoeff) * std::log(setup.restitutionCoeff) / 
    (math::pi * math::pi + std::log(setup.restitutionCoeff) * std::log(setup.restitutionCoeff))));

    setup.dampingNCoeff = -real_t(2.0) * std::sqrt(mij * setup.stiffnessCoeff) * (std::log(setup.restitutionCoeff) / 
    std::sqrt(math::pi * math::pi + (std::log(setup.restitutionCoeff) * std::log(setup.restitutionCoeff))));
    setup.dampingTCoeff = setup.dampingNCoeff;

    auto outputParameters = env.config()->getOneBlock("OutputParameters");
    setup.vtkBaseFolder = outputParameters.getParameter< std::string >("vtkBaseFolder");
    setup.vtkWriteFrequency = outputParameters.getParameter< uint_t >("vtkWriteFrequency");
    setup.logInfoFrequency = outputParameters.getParameter< uint_t >("logInfoFrequency");

    // check sanity of input parameters
    //setup.sanity_check();
    
    setup.printSetup();

    ////////////////////////////
    // simulation setup
    ////////////////////////////

    // simulation domain
    const auto domainSimulation = AABB(real_c(0.0), real_c(0.0), real_c(0.0),
    real_c(setup.domainSize[0]), real_c(setup.domainSize[1]), real_c(setup.domainSize[2]) );

    // create blockforect
    auto forest = blockforest::createBlockForest(domainSimulation, setup.numBlocks, setup.isPeriodic);

    //
    // PE setup
    //

    // generate IDs of specified PE body types
    pe::SetBodyTypeIDs< BodyTypeTuple >::execute();

    // add global body storage for PE bodies
    shared_ptr< pe::BodyStorage > globalBodyStorage = make_shared< pe::BodyStorage >();

    // add block-local body storage
    const auto bodyStorageID = forest->addBlockData(pe::createStorageDataHandling< BodyTypeTuple >(), "bodyStorage");

    // add data-handling for coarse collision dection
    const auto ccdID = forest->addBlockData(pe::ccd::createHashGridsDataHandling(globalBodyStorage, bodyStorageID), "ccd");

    // add data-handling for fine collision dection
    const auto fcdID = forest->addBlockData(pe::fcd::createGenericFCDDataHandling< BodyTypeTuple, pe::fcd::AnalyticCollideFunctor >(), "fcd");

    // add contact solver - DEM soft contacts
    const auto cr = make_shared< pe::cr::DEM >(globalBodyStorage, forest, bodyStorageID, ccdID, fcdID);

    // define particle material
    auto peMaterial = pe::createMaterial("particleMat", setup.density_ratio, setup.restitutionCoeff, 
    setup.frictionSCoeff, setup.frictionDCoeff, setup.poisson, setup.young, 
    setup.stiffnessCoeff, setup.dampingNCoeff, setup.dampingTCoeff);

    // create bounding planes
    // top and bottom - z
    pe::createPlane(*globalBodyStorage, 0, Vector3<real_t>(0, 0, -1), Vector3<real_t>(0, 0, domainSimulation.zMax()), peMaterial);
    pe::createPlane(*globalBodyStorage, 0, Vector3<real_t>(0, 0,  1), Vector3<real_t>(0, 0, domainSimulation.zMin()), peMaterial);
    // front and back - y
    pe::createPlane(*globalBodyStorage, 0, Vector3<real_t>(0,  1, 0), Vector3<real_t>(0, domainSimulation.yMin(), 0), peMaterial);
    pe::createPlane(*globalBodyStorage, 0, Vector3<real_t>(0, -1, 0), Vector3<real_t>(0, domainSimulation.yMax(), 0), peMaterial);
    // left and right - x - periodic
    pe::createPlane(*globalBodyStorage, 0, Vector3<real_t>( 1, 0, 0), Vector3<real_t>(domainSimulation.xMin(), 0, 0), peMaterial);
    pe::createPlane(*globalBodyStorage, 0, Vector3<real_t>(-1, 0, 0), Vector3<real_t>(domainSimulation.xMax(), 0, 0), peMaterial);

    // generate initial spherical particles
    createMonolayerSpheresSC(*forest, *globalBodyStorage, bodyStorageID, setup, domainSimulation, peMaterial);

    // sync
    pe::syncShadowOwners< BodyTypeTuple >(*forest, bodyStorageID);

    //
    // Output setup
    //

    // add vtk output for the domain decomposition
    vtk::writeDomainDecomposition(forest);

    // add vtk output for particles
    const auto bodyVTKOutput = make_shared< pe::SphereVtkOutput >(bodyStorageID, *forest);
    const auto bodyVTKWriter = vtk::createVTKOutput_PointData(bodyVTKOutput, "bodies", setup.vtkWriteFrequency, setup.vtkBaseFolder);

    /////////////////////////
    //  Time loop
    /////////////////////////

    for (uint_t timestep = uint_c(0); timestep < setup.timesteps; ++timestep)
    {
        // log information
        if (timestep % setup.logInfoFrequency == uint_c(0))
        {
            WALBERLA_LOG_INFO_ON_ROOT("timestep: " << timestep);
        }

        // perform a PE timestep
        cr->timestep(setup.dt);

        // sync
        //pe::sync
    }

    return EXIT_SUCCESS;
}

} //namespace spinners_suspension_dpm

int main( int argc, char **argv){
    return spinners_suspension::main(argc, argv);
}