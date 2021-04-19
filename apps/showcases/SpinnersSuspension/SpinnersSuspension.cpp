#include "core/all.h"

#include "blockforest/Initialization.h"
#include "blockforest/communication/all.h"

#include "core/logging/all.h"

#include "pe/basic.h"
#include "pe/Types.h"
#include "pe/synchronization/SyncShadowOwners.h"
#include "pe/utility/DestroyBody.h"
#include "pe/vtk/SphereVtkOutput.h"

#include "lbm/boundary/all.h"
#include "lbm/communication/PdfFieldPackInfo.h"
#include "lbm/field/AddToStorage.h"
#include "lbm/field/PdfField.h"
#include "lbm/lattice_model/D2Q9.h"
#include "lbm/sweeps/CellwiseSweep.h"
#include "lbm/sweeps/SweepWrappers.h"
#include "lbm/vtk/all.h"

#include "vtk/VTKOutput.h"

#include <tuple>

namespace spinners_suspension
{
// using
using namespace walberla;
using walberla::uint_t;

// typedefs
using LatticeModel_T = lbm::D2Q9<lbm::collision_model::TRT, false>;
const real_t magicNumberTRT = lbm::collision_model::TRT::threeSixteenth;

using Stencil_T = LatticeModel_T::Stencil;
using PdfField_T = lbm::PdfField<LatticeModel_T>;

using flag_t = walberla::uint8_t;
using FlagField_T = FlagField<flag_t>;
using BodyField_T = GhostLayerField<pe::BodyID, 1>;

const uint_t FieldGhostLayers = 1;

// boundary handling


// body types
using BodyTypeTuple = std::tuple< pe::Plane, pe::Sphere>;

// flags

const FlagUID Fluid_Flag ( "fluid" );

const FlagUID MEM_Flag ("moving obstacle");

const FlagUID FormerMEM_Flag ( "former moving obstacle" );

/////////////////////
// parameters
/////////////////////
struct Setup
{
    // domain parameters
    Vector3< uint_t > numBlocks; // number of blocks in x,y,z direction
    Vector3< uint_t > domainSize; // domian size in x,y,z direction
    Vector3< bool > isPeriodic; // whether periodic in x,y,z direction
    bool boundX; // bounding walls in x-axis
    bool boundY; // bounding walls in y-axis
    bool boundZ; // bounding walls in z-axis

    // simulation parameters
    uint_t timesteps; // simulation time steps
    real_t dt; // time interval
    real_t dx;

    uint_t substepsPE; // number of PE calls in each subcycle

    // fluid properties
    real_t fluid_density; // density of fluid
    real_t viscosity; // viscosity of fluid
    real_t omega;

    // particle properties
    real_t particle_density; // density of particles

    uint_t numParticles; // number of particles

    uint_t particle_number_1; // number of type 1 particles
    uint_t particle_number_2; // number of type 2 particles
    real_t particle_diameter_1; // particle diameter of first type
    real_t particle_diameter_2; // particle diameter of second type
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

    real_t density_ratio;

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

        WALBERLA_CHECK(particle_diameter_1 > 0 && particle_diameter_2 > 0, "particle diamters should be positive.");
    }
};

/////////////////////////
// auxiliary functions
/////////////////////////
uint_t generateRandomSpheres(StructuredBlockForest & forest, pe::BodyStorage & globalBodyStorage, const BlockDataID & bodyStorageID,
                                const Setup & setup, const AABB & domainGeneration, pe::MaterialID & material);

class ForceOnBodiesAdder
{
public:

   ForceOnBodiesAdder( const shared_ptr<StructuredBlockStorage> & blockStorage, const BlockDataID & bodyStorageID,
                       const Vector3<real_t> & forcePerVolume )
         : blockStorage_( blockStorage ), bodyStorageID_( bodyStorageID ), forcePerVolume_( forcePerVolume )
   { }

   // set a force on all (only local, to avoid force duplication) bodies
   void operator()()
   {
      for( auto blockIt = blockStorage_->begin(); blockIt != blockStorage_->end(); ++blockIt )
      {
         for( auto bodyIt = pe::LocalBodyIterator::begin( *blockIt, bodyStorageID_); bodyIt != pe::LocalBodyIterator::end(); ++bodyIt )
         {
            real_t volume = bodyIt->getVolume();
            bodyIt->addForce ( forcePerVolume_ * volume );
         }
      }
   }

private:

   shared_ptr<StructuredBlockStorage> blockStorage_;
   const BlockDataID bodyStorageID_;
   Vector3<real_t> forcePerVolume_;
};

class CollisionPropertiesEvaluator
{
public:
   CollisionPropertiesEvaluator( pe::cr::ICR & collisionResponse ) : collisionResponse_( collisionResponse ), maximumPenetration_(real_t(0))
   {}

   void operator()()
   {
      real_t maxPen = collisionResponse_.getMaximumPenetration();
      maximumPenetration_ = std::max( maximumPenetration_, maxPen );
      WALBERLA_MPI_SECTION()
      {
         mpi::allReduceInplace( maximumPenetration_, mpi::MAX );
      }
   }
   real_t getMaximumPenetration()
   {
      return maximumPenetration_;
   }
   void resetMaximumPenetration()
   {
      maximumPenetration_ = real_t(0);
   }
private:
   pe::cr::ICR & collisionResponse_;
   real_t maximumPenetration_;
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
    // Parse parameters
    //////////////////////////////////

    auto domainParameters = env.config()->getOneBlock("DomainParameters");
    setup.numBlocks = domainParameters.getParameter< Vector3< uint_t > >("numBlocks");
    setup.domainSize = domainParameters.getParameter< Vector3< uint_t > >("domainSize");
    setup.isPeriodic = domainParameters.getParameter < Vector3<  bool > >("isPeriodic");
    setup.boundX = domainParameters.getParameter< bool >("boundX");
    setup.boundY = domainParameters.getParameter< bool >("boundY");
    setup.boundZ = domainParameters.getParameter< bool >("boundZ");

    auto simulationParameters = env.config()->getOneBlock("SimulationParameters");
    setup.timesteps = simulationParameters.getParameter< uint_t >("timesteps");
    setup.dt = simulationParameters.getParameter< real_t >("dt");
    setup.dx = simulationParameters.getParameter< real_t >("dx");
    setup.substepsPE = simulationParameters.getParameter< uint_t >("substepsPE");

    auto fluidProperties = env.config()->getOneBlock("FluidProperties");
    setup.fluid_density = fluidProperties.getParameter< real_t >("fluid_density");
    setup.viscosity = fluidProperties.getParameter< real_t >("viscosity");
    setup.omega = lbm::collision_model::omegaFromViscosity(setup.viscosity);

    auto particleProperties = env.config()->getOneBlock("ParticleProperties");
    setup.particle_density = particleProperties.getParameter< real_t >("particle_density");
    setup.fluid_density = particleProperties.getParameter< real_t >("fluid_density");
    setup.density_ratio = setup.particle_density / setup.fluid_density;

    setup.particle_number_1 = particleProperties.getParameter< uint_t >("particle_number_1");
    setup.particle_number_2 = particleProperties.getParameter< uint_t >("particle_number_2");
    setup.particle_diameter_1 = particleProperties.getParameter< real_t >("particle_diameter_1");
    setup.particle_diameter_2 = particleProperties.getParameter< real_t >("particle_diameter_2");
    setup.particle_diameter_max = std::max(setup.particle_diameter_1, setup.particle_diameter_2);
    setup.particle_diameter_avg = (setup.particle_diameter_1 + setup.particle_diameter_2) / real_t(2.0);
    setup.particle_volume_avg = setup.particle_diameter_avg * setup.particle_diameter_avg * setup.particle_diameter_avg * math::pi / real_t(6.0);
    setup.particle_mass_avg = setup.density_ratio * setup.particle_volume_avg;

    // collision related material properties
    setup.restitutionCoeff = particleProperties.getParameter< real_t >("restitutionCoeff");
    setup.frictionSCoeff = particleProperties.getParameter< real_t >("frictionSCoeff");
    setup.frictionDCoeff = particleProperties.getParameter< real_t >("frictionDCoeff");
    setup.poisson = particleProperties.getParameter< real_t >("poisson");
    setup.young = particleProperties.getParameter< real_t >("young");
    setup.contactT = particleProperties.getParameter< real_t >("contactT");

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
    const uint_t XBlocks = setup.numBlocks[0];
    const uint_t YBlocks = setup.numBlocks[1];
    const uint_t ZBlocks = setup.numBlocks[2];

    const uint_t Lx = setup.domainSize[0];
    const uint_t Ly = setup.domainSize[1];
    const uint_t Lz = setup.domainSize[2];

    const bool xPeriodic = setup.isPeriodic[0];
    const bool yPeriodic = setup.isPeriodic[1];
    const bool zPeriodic = setup.isPeriodic[2];

    const uint_t XCells = Lx / XBlocks;
    const uint_t YCells = Ly / YBlocks;
    const uint_t ZCells = Lz / ZBlocks;

    if (XBlocks * XCells != Lx || YBlocks * YCells != Ly || ZBlocks * ZCells != Lz) WALBERLA_ABORT("Domain Decomposition Failed.");

    const auto domainSimulation = AABB(real_c(0.0), real_c(0.0), real_c(0.0),
    real_c(setup.domainSize[0]), real_c(setup.domainSize[1]), real_c(setup.domainSize[2]) );

    // create blockforect
    //auto forest = blockforest::createBlockForest(domainSimulation, setup.numBlocks, setup.isPeriodic);
    auto blocks = blockforest::createUniformBlockGrid(XBlocks, YBlocks, ZBlocks, 
                                                      XCells, YCells, ZCells,
                                                      setup.dx,
                                                      0, false, false,
                                                      xPeriodic, yPeriodic, zPeriodic,
                                                      false);

    //--------------
    // PE setup
    //--------------

    // generate IDs of specified PE body types
    pe::SetBodyTypeIDs< BodyTypeTuple >::execute();

    // add global body storage for PE bodies
    shared_ptr< pe::BodyStorage > globalBodyStorage = make_shared< pe::BodyStorage >();

    // add block-local body storage
    const auto bodyStorageID = blocks->addBlockData(pe::createStorageDataHandling< BodyTypeTuple >(), "bodyStorage");

    // add data-handling for coarse collision dection
    const auto ccdID = blocks->addBlockData(pe::ccd::createHashGridsDataHandling(globalBodyStorage, bodyStorageID), "ccd");

    // add data-handling for fine collision dection
    const auto fcdID = blocks->addBlockData(pe::fcd::createGenericFCDDataHandling< BodyTypeTuple, pe::fcd::AnalyticCollideFunctor >(), "fcd");

    // add contact solver - using DEM soft contacts
    const auto cr = make_shared< pe::cr::DEM >(globalBodyStorage, blocks->getBlockStoragePointer(), bodyStorageID, ccdID, fcdID);

    // define particle material
    auto peMaterial = pe::createMaterial("particleMat", setup.density_ratio, setup.restitutionCoeff, 
    setup.frictionSCoeff, setup.frictionDCoeff, setup.poisson, setup.young, 
    setup.stiffnessCoeff, setup.dampingNCoeff, setup.dampingTCoeff);

    WALBERLA_LOG_INFO_ON_ROOT("Summary of particle material properties:\n"
                              << " - coefficient of restitution = " << setup.restitutionCoeff << "\n"
                              << " - coefficient of static friction = " << setup.frictionSCoeff << "\n"
                              << " - coefficient of dynamic friction = " << setup.frictionDCoeff << "\n"
                              << " - stiffness coefficient kn = " << setup.stiffnessCoeff << "\n"
                              << " - normal damping coefficient = " << setup.dampingNCoeff << "\n"
                              << " - tangential damping coefficient = " << setup.dampingTCoeff << "\n"
                              << " - contact time Tc = " << setup.contactT);

    // create bounding walls
    // top and bottom - z
    if (setup.boundZ)
    {
        pe::createPlane(*globalBodyStorage, 0, Vector3<real_t>(0, 0, -1), Vector3<real_t>(0, 0, domainSimulation.zMax()), peMaterial);
        pe::createPlane(*globalBodyStorage, 0, Vector3<real_t>(0, 0,  1), Vector3<real_t>(0, 0, domainSimulation.zMin()), peMaterial);
        WALBERLA_LOG_INFO_ON_ROOT("Bounding walls in the z direction created.");
    }
    // front and back - y
    if (setup.boundY)
    {
        pe::createPlane(*globalBodyStorage, 0, Vector3<real_t>(0,  1, 0), Vector3<real_t>(0, domainSimulation.yMin(), 0), peMaterial);
        pe::createPlane(*globalBodyStorage, 0, Vector3<real_t>(0, -1, 0), Vector3<real_t>(0, domainSimulation.yMax(), 0), peMaterial);
        WALBERLA_LOG_INFO_ON_ROOT("Bounding walls in the y direction created.");
    }
    // left and right - x
    if (setup.boundX)
    {
        pe::createPlane(*globalBodyStorage, 0, Vector3<real_t>( 1, 0, 0), Vector3<real_t>(domainSimulation.xMin(), 0, 0), peMaterial);
        pe::createPlane(*globalBodyStorage, 0, Vector3<real_t>(-1, 0, 0), Vector3<real_t>(domainSimulation.xMax(), 0, 0), peMaterial);
        WALBERLA_LOG_INFO_ON_ROOT("Bounding walls in the x direction created.");
    }

    // set up synchronization procedure
    const real_t overlap = real_c( 1.5 ) * setup.dx;
    std::function<void(void)> syncCall;
    if ( XBlocks <= uint_t(4) )
        syncCall = std::bind( pe::syncNextNeighbors<BodyTypeTuple>, std::ref(blocks->getBlockForest()), bodyStorageID, static_cast<WcTimingTree*>(nullptr), overlap, false);
    else
        syncCall = std::bind( pe::syncShadowOwners<BodyTypeTuple>, std::ref(blocks->getBlockForest()), bodyStorageID, static_cast<WcTimingTree*>(nullptr), overlap, false);

    ////////////////////////////
    // Initialization
    ////////////////////////////

    // generate initial spherical particles

    real_t layer_thickness = setup.dx;
    real_t layer_zpos = real_c(0.5) * setup.dx;
    real_t radius_max = setup.particle_diameter_max / real_c(2.0);
    const auto domainGeneration = AABB(domainSimulation.xMin() + radius_max, domainSimulation.yMin() + radius_max, domainSimulation.zMin(), 
                                       domainSimulation.xMax() - radius_max, domainSimulation.yMax() - radius_max, domainSimulation.zMin() + layer_thickness);

    // collision properties evaluator
    auto OverlapEvaluator = make_shared<CollisionPropertiesEvaluator>(*cr);

    // random generation of spherical particles
    setup.numParticles = generateRandomSpheresLayer(*blocks, *globalBodyStorage, bodyStorageID, setup, domainGeneration, peMaterial, layer_zpos);
    // sync the created particles between processes
    syncCall();

    WALBERLA_LOG_INFO_ON_ROOT(setup.numParticles << " spheres created.");

    //
    // Add data to blocks
    //

    // create the lattice model
    LatticeModel_T latticeModel = LatticeModel_T( lbm::collision_model::TRT::constructWithMagicNumber( setup.omega, magicNumberTRT));

    // add pdf field
    BlockDataID pdfFieldID = lbm::addPdfFieldToStorage< LatticeModel_T >( blocks, "pdf field (zyxf)", latticeModel, uInfty, rhoInit, FieldGhostLayers, field::zyxf);

    //-------------
    // Output setup
    //-------------

    // add vtk output for the domain decomposition
    vtk::writeDomainDecomposition(blocks, "domain_decomposition", setup.vtkBaseFolder);

    // add vtk output for particles
    const auto bodyVTKOutput = make_shared< pe::SphereVtkOutput >(bodyStorageID, blocks->getBlockStorage());
    const auto bodyVTKWriter = vtk::createVTKOutput_PointData(bodyVTKOutput, "bodies", setup.vtkWriteFrequency, setup.vtkBaseFolder);

    // resolve particle overlaps
    const uint_t initialPEsteps = uint_c(100000);
    const real_t dt_DEM_init = setup.contactT / real_t(uint_c(10) * setup.substepsPE);
    const real_t overlapTarget = real_t(0.01) * setup.particle_diameter_avg;

    // temperary bounding plane
    //auto boundingPlane = pe::createPlane(*globalBodyStorage, 1, Vector3<real_t>(0, 0, -1), Vector3<real_t>(0, 0, domainGeneration.zMax() + radius_max), peMaterial);

    for (uint_t pestep = uint_c(0); pestep < initialPEsteps; ++pestep)
    {
        cr->timestep(dt_DEM_init);
        pe::syncShadowOwners< BodyTypeTuple >(blocks->getBlockForest(), bodyStorageID);

        OverlapEvaluator->operator()();

        real_t maxOverlap = OverlapEvaluator->getMaximumPenetration();

        if (maxOverlap < overlapTarget)
        {
            WALBERLA_LOG_INFO_ON_ROOT("Carried out " << pestep << " PE-only steps to resolve initial overlaps.");
            break;
        }
        else
        {
            if (pestep % setup.logInfoFrequency == uint_c(0))
            {
                WALBERLA_LOG_INFO_ON_ROOT("timestep: " << pestep << " - current max overlap = " << maxOverlap / setup.particle_diameter_avg * real_c(100) << "%");
            }

            if (pestep % setup.vtkWriteFrequency == uint_c(0))
            {
                bodyVTKWriter->write();
            }
        }

        OverlapEvaluator->resetMaximumPenetration();
    }

    // destroy temperary bounding plane
    //pe::destroyBodyBySID(*globalBodyStorage, blocks->getBlockStorage(), bodyStorageID, boundingPlane->getSystemID());

    // comunication
    pe::syncShadowOwners< BodyTypeTuple >(blocks->getBlockForest(), bodyStorageID);
    bodyVTKWriter->write();

    /////////////////////////
    //  Time loop
    /////////////////////////

    return EXIT_SUCCESS;
}

uint_t generateRandomSpheresLayer(StructuredBlockForest & forest, pe::BodyStorage & globalBodyStorage, const BlockDataID & bodyStorageID,
                                const Setup & setup, const AABB & domainGeneration, pe::MaterialID & material, real_t layer_zpos)
{
    //Randomly generate certain number of bidisperse spheres inside a specified domain

    uint_t N1 = 0;
    uint_t N2 = 0;

    real_t xpos, ypos, zpos, diameter;

    WALBERLA_LOG_INFO_ON_ROOT("Creating " << setup.particle_number_1 << " type 1 spheres with diameter = " << setup.particle_diameter_1);

    while (N1 < setup.particle_number_1)
    {
        WALBERLA_ROOT_SECTION()
        {
            xpos = math::realRandom<real_t>(domainGeneration.xMin(), domainGeneration.xMax());
            ypos = math::realRandom<real_t>(domainGeneration.yMin(), domainGeneration.yMax());
            //zpos = math::realRandom<real_t>(domainGeneration.zMin(), domainGeneration.zMax());
            zpos = layer_zpos;
            diameter = setup.particle_diameter_1;
        }

        WALBERLA_MPI_SECTION()
        {
            mpi::broadcastObject(xpos);
            mpi::broadcastObject(ypos);
            mpi::broadcastObject(zpos);
            mpi::broadcastObject(diameter);
        }

        pe::SphereID sp = pe::createSphere( globalBodyStorage, forest.getBlockStorage(), bodyStorageID, 0, Vector3<real_t>(xpos,ypos,zpos), diameter * real_c(0.5), material);

        if (sp != nullptr) ++N1;
    }

    WALBERLA_LOG_INFO_ON_ROOT("Creating " << setup.particle_number_2 << " type 2 spheres with diameter = " << setup.particle_diameter_2);

    while (N2 < setup.particle_number_2)
    {
        WALBERLA_ROOT_SECTION()
        {
            xpos = math::realRandom<real_t>(domainGeneration.xMin(), domainGeneration.xMax());
            ypos = math::realRandom<real_t>(domainGeneration.yMin(), domainGeneration.yMax());
            //zpos = math::realRandom<real_t>(domainGeneration.zMin(), domainGeneration.zMax());
            zpos = layer_zpos;
            diameter = setup.particle_diameter_2;
        }

        WALBERLA_MPI_SECTION()
        {
            mpi::broadcastObject(xpos);
            mpi::broadcastObject(ypos);
            mpi::broadcastObject(zpos);
            mpi::broadcastObject(diameter);
        }

        pe::SphereID sp = pe::createSphere( globalBodyStorage, forest.getBlockStorage(), bodyStorageID, 0, Vector3<real_t>(xpos,ypos,zpos), diameter * real_c(0.5), material);

        if (sp != nullptr) ++N2;
    }

    return N1 + N2;
}

} //namespace spinners_suspension_dpm

int main( int argc, char **argv){
    return spinners_suspension::main(argc, argv);
}