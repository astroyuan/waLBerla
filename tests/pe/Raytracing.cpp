#include "pe/utility/BodyCast.h"

#include "pe/Materials.h"

#include "pe/rigidbody/Box.h"
#include "pe/rigidbody/Capsule.h"
#include "pe/rigidbody/Sphere.h"
#include "pe/rigidbody/Plane.h"
#include "pe/rigidbody/Union.h"

#include "pe/rigidbody/SetBodyTypeIDs.h"
#include "pe/Types.h"

#include "core/debug/TestSubsystem.h"
#include "core/DataTypes.h"
#include "core/math/Vector3.h"

#include <pe/raytracing/Ray.h>
#include <pe/raytracing/Intersects.h>

using namespace walberla;
using namespace walberla::pe;
using namespace walberla::pe::raytracing;

typedef boost::tuple<Box, Capsule, Plane, Sphere> BodyTuple ;

void SphereIntersectsTest()
{
   MaterialID iron = Material::find("iron");
   Sphere sp1(123, 1, Vec3(3,3,3), Vec3(0,0,0), Quat(), 2, iron, false, true, false);
   real_t t;
   
   // ray through the center
   Ray ray1(Vec3(-5,3,3), Vec3(1,0,0));
   WALBERLA_LOG_INFO("RAY -> SPHERE");
   
   WALBERLA_CHECK(intersects(&sp1, &ray1, &t));
   WALBERLA_CHECK(realIsEqual(t, real_t(6)))

   // ray tangential
   Ray ray2(Vec3(-5,3,3), Vec3(7.5,0,real_t(std::sqrt(real_t(15))/real_t(2))));
   WALBERLA_CHECK(!intersects(&sp1, &ray2, &t));
   
   // sphere behind ray origin
   Sphere sp2(123, 1, Vec3(-8,3,3), Vec3(0,0,0), Quat(), 2, iron, false, true, false);
   WALBERLA_CHECK(!intersects(&sp2, &ray1, &t));
   
   // sphere around ray origin
   Sphere sp3(123, 1, Vec3(-5,3,3), Vec3(0,0,0), Quat(), 2, iron, false, true, false);
   WALBERLA_CHECK(intersects(&sp3, &ray1, &t));
   WALBERLA_CHECK(realIsEqual(t, real_t(2)));
}

void PlaneIntersectsTest() {
   MaterialID iron = Material::find("iron");
   // plane with center 3,3,3 and parallel to y-z plane
   Plane pl1(1, 1, Vec3(3, 3, 3), Vec3(1, 0, 0), real_t(1.0), iron);
   
   Ray ray1(Vec3(-5,3,3), Vec3(1,0,0));
   real_t t;
   
   WALBERLA_LOG_INFO("RAY -> PLANE");
   WALBERLA_CHECK(intersects(&pl1, &ray1, &t), "ray through center did not hit");
   WALBERLA_CHECK(realIsEqual(t, real_t(8)), "distance between ray and plane is incorrect");
   
   Ray ray2(Vec3(-5,3,3), Vec3(1,0,-1));
   WALBERLA_CHECK(intersects(&pl1, &ray2, &t), "ray towards random point on plane didn't hit");
   WALBERLA_CHECK(realIsEqual(t, real_t(sqrt(real_t(128)))), "distance between ray and plane is incorrect");

   // plane with center 3,3,3 and parallel to x-z plane
   Plane pl2(1, 1, Vec3(3, 3, 3), Vec3(0, 1, 0), real_t(1.0), iron);
   WALBERLA_CHECK(!intersects(&pl2, &ray1, &t), "ray parallel to plane shouldnt hit");
   
   // plane with center -10,3,3 and parallel to y-z plane
   Plane pl4(1, 1, Vec3(-10, 3, 3), Vec3(1, 0, 0), real_t(1.0), iron);
   WALBERLA_CHECK(!intersects(&pl4, &ray1, &t), "ray hit plane behind origin");
}

void BoxIntersectsTest() {
   MaterialID iron = Material::find("iron");
   Box box1(127, 5, Vec3(-15, 0, 0), Vec3(0, 0, 0), Quat(), Vec3(10, 10, 10), iron, false, true, false);
   
   Ray ray1(Vec3(-5,3,3), Vec3(1,0,0));
   real_t t;
   
   WALBERLA_LOG_INFO("RAY -> BOX");

   WALBERLA_CHECK(!intersects(&box1, &ray1, &t));
   
   Box box2(128, 5, Vec3(-2, 0, 0), Vec3(0, 0, 0), Quat(), Vec3(10, 10, 10), iron, false, true, false);
   WALBERLA_CHECK(intersects(&box2, &ray1, &t));
   WALBERLA_CHECK(realIsEqual(t, real_t(8), 1e-7));
   
   Box box3(128, 5, Vec3(5, 0, 0), Vec3(0, 0, 0), Quat(), Vec3(10, 10, 10), iron, false, true, false);
   WALBERLA_CHECK(intersects(&box3, &ray1, &t));
   WALBERLA_CHECK(realIsEqual(t, real_t(5)));
   
   Ray ray2(Vec3(-2,0,0), Vec3(1,0,1));
   WALBERLA_LOG_INFO(intersects(&box3, &ray2, &t));
   WALBERLA_LOG_INFO(t);
   
   Ray ray3(Vec3(-5,3,3), Vec3(2, -1.5, 0.5));
   Box box4(128, 5, Vec3(8, 0, 0), Vec3(0, 0, 0), Quat(), Vec3(10, 10, 10), iron, false, true, false);
   WALBERLA_LOG_INFO("-- -- -- --");
   WALBERLA_LOG_INFO(intersects(&box4, &ray3, &t));
   WALBERLA_LOG_INFO("t: " << t);
   
   Ray ray4(Vec3(-5,3,3), Vec3(2, -1, 0.5));
   WALBERLA_LOG_INFO("-- -- -- --");
   WALBERLA_LOG_INFO(intersects(&box4, &ray4, &t));
   WALBERLA_LOG_INFO("t: " << t);
}

int main( int argc, char** argv )
{
   walberla::debug::enterTestMode();
   walberla::MPIManager::instance()->initializeMPI( &argc, &argv );
   
   SetBodyTypeIDs<BodyTuple>::execute();
   
   SphereIntersectsTest();
   PlaneIntersectsTest();
   //BoxIntersectsTest(); // does not work yet
   
   return EXIT_SUCCESS;
}
