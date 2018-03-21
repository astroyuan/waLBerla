//======================================================================================================================
//
//  This file is part of waLBerla. waLBerla is free software: you can
//  redistribute it and/or modify it under the terms of the GNU General Public
//  License as published by the Free Software Foundation, either version 3 of
//  the License, or (at your option) any later version.
//
//  waLBerla is distributed in the hope that it will be useful, but WITHOUT
//  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
//  FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
//  for more details.
//
//  You should have received a copy of the GNU General Public License along
//  with waLBerla (see COPYING.txt). If not, see <http://www.gnu.org/licenses/>.
//
//! \file Raytracer.cpp
//! \author Lukas Werner
//
//======================================================================================================================

#include "Raytracer.h"
#include "geometry/structured/extern/lodepng.h"

namespace walberla {
namespace pe {
namespace raytracing {
   
#ifdef WALBERLA_BUILD_WITH_MPI
void BodyIntersectionInfo_Comparator_MPI_OP( BodyIntersectionInfo *in, BodyIntersectionInfo *inout, int *len, MPI_Datatype *dptr) {
   WALBERLA_UNUSED(dptr);
   for (int i = 0; i < *len; ++i) {
      if (in->bodySystemID != 0 && inout->bodySystemID != 0) {
         WALBERLA_ASSERT(in->imageX == inout->imageX && in->imageY == inout->imageY, "coordinates of infos do not match: " << in->imageX << "/" << in->imageY << " and " << inout->imageX << "/" << inout->imageY);
      }
      
      if ((in->t < inout->t && in->bodySystemID != 0) || (inout->bodySystemID == 0 && in->bodySystemID != 0)) {
         // info in "in" is closer than the one in "inout" -> update inout to values of in
         inout->imageX = in->imageX;
         inout->imageY = in->imageY;
         inout->bodySystemID = in->bodySystemID;
         inout->t = in->t;
         inout->r = in->r;
         inout->g = in->g;
         inout->b = in->b;
      }
      
      in++;
      inout++;
   }
}
#endif

/*!\brief Instantiation constructor for the Raytracer class.
 *
 * \param forest BlockForest the raytracer operates on.
 * \param storageID Storage ID of the block data storage the raytracer operates on.
 * \param pixelsHorizontal Horizontal amount of pixels of the generated image.
 * \param pixelsVertical Vertical amount of pixels of the generated image.
 * \param fov_vertical Vertical field-of-view of the camera.
 * \param cameraPosition Position of the camera in the global world frame.
 * \param lookAtPoint Point the camera looks at in the global world frame.
 * \param upVector Vector indicating the upwards direction of the camera.
 * \param backgroundColor Background color of the scene.
 * \param blockAABBIntersectionPadding The padding applied in block AABB intersection pretesting. Usually not required.
 *                                     Set it to the value of the farthest distance a object might protrude from
 *                                     its containing block.
 */
Raytracer::Raytracer(const shared_ptr<BlockStorage> forest, const BlockDataID storageID,
                     const shared_ptr<BodyStorage> globalBodyStorage,
                     const BlockDataID ccdID,
                     uint16_t pixelsHorizontal, uint16_t pixelsVertical,
                     real_t fov_vertical, uint8_t antiAliasFactor,
                     const Vec3& cameraPosition, const Vec3& lookAtPoint, const Vec3& upVector,
                     const Lighting& lighting,
                     const Color& backgroundColor,
                     real_t blockAABBIntersectionPadding,
                     std::function<ShadingParameters (const BodyID)> bodyToShadingParamsFunction)
   : forest_(forest), storageID_(storageID), globalBodyStorage_(globalBodyStorage), ccdID_(ccdID),
   pixelsHorizontal_(pixelsHorizontal), pixelsVertical_(pixelsVertical),
   fov_vertical_(fov_vertical), antiAliasFactor_(antiAliasFactor),
   cameraPosition_(cameraPosition), lookAtPoint_(lookAtPoint), upVector_(upVector),
   lighting_(lighting),
   backgroundColor_(backgroundColor),
   blockAABBIntersectionPadding_(blockAABBIntersectionPadding),
   tBufferOutputEnabled_(false),
   tBufferOutputDirectory_("."),
   imageOutputEnabled_(true),
   localImageOutputEnabled_(false),
   imageOutputDirectory_("."),
   filenameTimestepWidth_(5),
   bodyToShadingParamsFunction_(bodyToShadingParamsFunction),
   raytracingAlgorithm_(RAYTRACE_HASHGRIDS),
   reductionMethod_(MPI_REDUCE) {
   
   setupView_();
   setupFilenameRankWidth_();
   setupMPI_();
}

/*!\brief Instantiation constructor for the Raytracer class using a config object for view setup.
 *
 * \param forest BlockForest the raytracer operates on.
 * \param storageID Storage ID of the block data storage the raytracer operates on.
 * \param config Config block for the raytracer.
 *
 * The config block has to contain image_x (int), image_y (int), fov_vertical (real, in degrees) and
 * antiAliasFactor (uint, between 1 and 4). Additionally a vector of reals for each of cameraPosition, lookAt
 * and the upVector. Optional is blockAABBIntersectionPadding (real) and backgroundColor (Vec3).
 * To output both process local and global tbuffers after raytracing, set tbuffer_output_directory (string).
 * For image output after raytracing, set image_output_directory (string); for local image output additionally set
 * local_image_output_enabled (bool) to true. outputFilenameTimestepZeroPadding (int) sets zero padding
 * for timesteps of output filenames.
 * For the lighting a config block named Lighting has to be defined, information about its contents is in Lighting.h.
 */
Raytracer::Raytracer(const shared_ptr<BlockStorage> forest, const BlockDataID storageID,
                     const shared_ptr<BodyStorage> globalBodyStorage,
                     const BlockDataID ccdID,
                     const Config::BlockHandle& config,
                     std::function<ShadingParameters (const BodyID)> bodyToShadingParamsFunction)
   : forest_(forest), storageID_(storageID), globalBodyStorage_(globalBodyStorage), ccdID_(ccdID),
   bodyToShadingParamsFunction_(bodyToShadingParamsFunction),
   raytracingAlgorithm_(RAYTRACE_HASHGRIDS),
   reductionMethod_(MPI_REDUCE) {
   WALBERLA_CHECK(config.isValid(), "No valid config passed to raytracer");
   
   pixelsHorizontal_ = config.getParameter<uint16_t>("image_x");
   pixelsVertical_ = config.getParameter<uint16_t>("image_y");
   fov_vertical_ = config.getParameter<real_t>("fov_vertical");
   antiAliasFactor_ = config.getParameter<uint8_t>("antiAliasFactor", 1);
   
   if (config.isDefined("tbuffer_output_directory")) {
      setTBufferOutputEnabled(true);
      setTBufferOutputDirectory(config.getParameter<std::string>("tbuffer_output_directory", "."));
      WALBERLA_LOG_INFO_ON_ROOT("t buffers will be written to " << getTBufferOutputDirectory() << ".");
   } else {
      setTBufferOutputEnabled(false);
   }
   
   setLocalImageOutputEnabled(config.getParameter<bool>("local_image_output_enabled", false));
      
   if (config.isDefined("image_output_directory")) {
      setImageOutputEnabled(true);
      setImageOutputDirectory(config.getParameter<std::string>("image_output_directory", "."));
      WALBERLA_LOG_INFO_ON_ROOT("Images will be written to " << getImageOutputDirectory() << ".");
   } else if (getLocalImageOutputEnabled()) {
      WALBERLA_ABORT("Cannot enable local image output without image_output_directory parameter being set.");
   }
      
   filenameTimestepWidth_ = config.getParameter<uint8_t>("filenameTimestepWidth", uint8_t(5));
   
   cameraPosition_ = config.getParameter<Vec3>("cameraPosition");
   lookAtPoint_ = config.getParameter<Vec3>("lookAt");
   upVector_ = config.getParameter<Vec3>("upVector");
   lighting_ = Lighting(config.getBlock("Lighting"));
   backgroundColor_ = config.getParameter<Color>("backgroundColor", Vec3(real_t(0.1), real_t(0.1), real_t(0.1)));

   blockAABBIntersectionPadding_ = config.getParameter<real_t>("blockAABBIntersectionPadding", real_t(0.0));

   std::string raytracingAlgorithm = config.getParameter<std::string>("raytracingAlgorithm", "RAYTRACE_HASHGRIDS");
   if (raytracingAlgorithm == "RAYTRACE_HASHGRIDS") {
      setRaytracingAlgorithm(RAYTRACE_HASHGRIDS);
   } else if (raytracingAlgorithm == "RAYTRACE_NAIVE") {
      setRaytracingAlgorithm(RAYTRACE_NAIVE);
   } else if (raytracingAlgorithm == "RAYTRACE_COMPARE_BOTH") {
      setRaytracingAlgorithm(RAYTRACE_COMPARE_BOTH);
   }
      
   std::string reductionMethod = config.getParameter<std::string>("reductionMethod", "MPI_REDUCE");
   if (reductionMethod == "MPI_REDUCE") {
      setReductionMethod(MPI_REDUCE);
   } else if (reductionMethod == "MPI_GATHER") {
      setReductionMethod(MPI_GATHER);
   }
      
   setupView_();
   setupFilenameRankWidth_();
   setupMPI_();
}

/*!\brief Utility function for setting up the view plane and calculating required variables.
 */
void Raytracer::setupView_() {
   // eye coordinate system setup
   n_ = (cameraPosition_ - lookAtPoint_).getNormalized();
   u_ = (upVector_ % n_).getNormalized();
   v_ = n_ % u_;
   
   // viewing plane setup
   d_ = (cameraPosition_ - lookAtPoint_).length();
   aspectRatio_ = real_t(pixelsHorizontal_) / real_t(pixelsVertical_);
   real_t fov_vertical_rad = fov_vertical_ * math::M_PI / real_t(180.0);
   viewingPlaneHeight_ = real_c(tan(fov_vertical_rad/real_t(2.))) * real_t(2.) * d_;
   viewingPlaneWidth_ = viewingPlaneHeight_ * aspectRatio_;
   viewingPlaneOrigin_ = lookAtPoint_ - u_*viewingPlaneWidth_/real_t(2.) - v_*viewingPlaneHeight_/real_t(2.);
   
   pixelWidth_ = viewingPlaneWidth_ / real_c(pixelsHorizontal_*antiAliasFactor_);
   pixelHeight_ = viewingPlaneHeight_ / real_c(pixelsVertical_*antiAliasFactor_);
}

/*!\brief Utility function for initializing the attribute filenameRankWidth.
 */
void Raytracer::setupFilenameRankWidth_() {
   WALBERLA_MPI_SECTION() {
      int numProcesses = mpi::MPIManager::instance()->numProcesses();
      filenameRankWidth_ = uint8_c(log10(numProcesses)+1);
   }
}

/*!\brief Utility function for setting up the MPI datatype and operation.
 */
void Raytracer::setupMPI_() {
#ifdef WALBERLA_BUILD_WITH_MPI
   MPI_Op_create((MPI_User_function *)BodyIntersectionInfo_Comparator_MPI_OP, true, &bodyIntersectionInfo_reduction_op);
   
   const int nblocks = 7;
   const int blocklengths[nblocks] = {1,1,1,1,1,1,1};
   MPI_Datatype types[nblocks] = {
      MPI_UNSIGNED, // for coordinate
      MPI_UNSIGNED, // for coordinate
      MPI_UNSIGNED_LONG_LONG, // for id
      MPI_DOUBLE, // for distance
      MPI_DOUBLE, // for color
      MPI_DOUBLE, // for color
      MPI_DOUBLE // for color
   };
   MPI_Aint displacements[nblocks];
   displacements[0] = offsetof(BodyIntersectionInfo, imageX);
   displacements[1] = offsetof(BodyIntersectionInfo, imageY);
   displacements[2] = offsetof(BodyIntersectionInfo, bodySystemID);
   displacements[3] = offsetof(BodyIntersectionInfo, t);
   displacements[4] = offsetof(BodyIntersectionInfo, r);
   displacements[5] = offsetof(BodyIntersectionInfo, g);
   displacements[6] = offsetof(BodyIntersectionInfo, b);
   
   MPI_Datatype tmp_type;
   MPI_Type_create_struct(nblocks, blocklengths, displacements, types, &tmp_type);
   
   MPI_Aint lb, extent;
   MPI_Type_get_extent( tmp_type, &lb, &extent );
   MPI_Type_create_resized( tmp_type, lb, extent, &bodyIntersectionInfo_mpi_type );
   
   MPI_Type_commit(&bodyIntersectionInfo_mpi_type);
#endif
}
   
/*!\brief Generates the filename for output files.
 * \param base String that precedes the timestap and rank info.
 * \param timestep Timestep this image is from.
 * \param isGlobalImage Whether this image is the fully stitched together one.
 */
std::string Raytracer::getOutputFilename(const std::string& base, size_t timestep, bool isGlobalImage) const {
   uint_t maxTimestep = uint_c(pow(10, filenameTimestepWidth_));
   WALBERLA_CHECK(timestep < maxTimestep, "Raytracer only supports outputting " << (maxTimestep-1) << " timesteps for the configured filename timestep width.");
   mpi::MPIRank rank = mpi::MPIManager::instance()->rank();
   std::stringstream fileNameStream;
   fileNameStream << base << "_";
   fileNameStream << std::setfill('0') << std::setw(int_c(filenameTimestepWidth_)) << timestep; // add timestep
   WALBERLA_MPI_SECTION() {
      // Appending the rank to the filename only makes sense if actually using MPI.
      fileNameStream << "+";
      if (isGlobalImage) {
         fileNameStream << "global";
      } else {
         fileNameStream << std::setfill('0') << std::setw(int_c(filenameRankWidth_)) << std::to_string(rank); // add rank
      }
   }
   fileNameStream << ".png"; // add extension
   return fileNameStream.str();
}

/*!\brief Writes the depth values of the intersections buffer to an image file in the tBuffer output directory.
 * \param intersectionsBuffer Buffer with intersection info for each pixel.
 * \param timestep Timestep this image is from.
 * \param isGlobalImage Whether this image is the fully stitched together one.
 */
void Raytracer::writeDepthsToFile(const std::vector<BodyIntersectionInfo>& intersectionsBuffer,
                                   size_t timestep, bool isGlobalImage) const {
   writeDepthsToFile(intersectionsBuffer, getOutputFilename("tbuffer", timestep, isGlobalImage));
}

/*!\brief Writes the depth values of the intersections buffer to an image file in the tBuffer output directory.
 * \param intersectionsBuffer Buffer with intersection info for each pixel.
 * \param fileName Name of the output file.
 */
void Raytracer::writeDepthsToFile(const std::vector<BodyIntersectionInfo>& intersectionsBuffer,
                                  const std::string& fileName) const {
   real_t inf = std::numeric_limits<real_t>::max();
   
   real_t t_max = 1;
   real_t t_min = inf;
   for (size_t x = 0; x < pixelsHorizontal_; x++) {
      for (size_t y = 0; y < pixelsVertical_; y++) {
         size_t i = coordinateToArrayIndex(x, y);
         real_t t = real_c(intersectionsBuffer[i].t);
         if (t < t_min) {
            t_min = t;
         }
      }
   }
   if (realIsIdentical(t_min, inf)) t_min = 0;
   
   t_max = forest_->getDomain().maxDistance(cameraPosition_);
   
   filesystem::path dir (getTBufferOutputDirectory());
   filesystem::path file (fileName);
   filesystem::path fullPath = dir / file;
   
   std::vector<u_char> lodeTBuffer(pixelsHorizontal_*pixelsVertical_);
   
   uint32_t l = 0;
   for (size_t y = pixelsVertical_-1; y > 0; y--) {
      for (size_t x = 0; x < pixelsHorizontal_; x++) {
         size_t i = coordinateToArrayIndex(x, y);
         u_char g = 0;
         real_t t = real_c(intersectionsBuffer[i].t);
         if (realIsIdentical(t, inf)) {
            g = (u_char)0;
         } else {
            real_t t_scaled = (1-(t-t_min)/(t_max-t_min));
            g = (u_char)(255 * std::max(std::min(t_scaled, real_t(1)), real_t(0)));
         }
         lodeTBuffer[l] = g;
         l++;
      }
   }
   
   uint32_t error = lodepng::encode(fullPath.string(), lodeTBuffer, getPixelsHorizontal(), getPixelsVertical(), LCT_GREY);
   if(error) {
      WALBERLA_LOG_WARNING("lodePNG error " << error << " when trying to save tbuffer file to " << fullPath.string() << ": " << lodepng_error_text(error));
   }
}

/*!\brief Writes the image of the current intersection buffer to a file in the image output directory.
 * \param intersectionsBuffer Buffer with intersection info for each pixel.
 * \param timestep Timestep this image is from.
 * \param isGlobalImage Whether this image is the fully stitched together one.
 */
void Raytracer::writeImageToFile(const std::vector<BodyIntersectionInfo>& intersectionsBuffer, size_t timestep,
                                 bool isGlobalImage) const {
   writeImageToFile(intersectionsBuffer, getOutputFilename("image", timestep, isGlobalImage));
}

/*!\brief Writes the image of the current intersection buffer to a file in the image output directory.
 * \param intersectionsBuffer Buffer with intersection info for each pixel.
 * \param fileName Name of the output file.
 */
void Raytracer::writeImageToFile(const std::vector<BodyIntersectionInfo>& intersectionsBuffer,
                                 const std::string& fileName) const {
   filesystem::path dir = getImageOutputDirectory();
   filesystem::path file (fileName);
   filesystem::path fullPath = dir / file;
   
   std::vector<u_char> lodeImageBuffer(pixelsHorizontal_*pixelsVertical_*3);
   
   uint32_t l = 0;
   real_t patchSize = real_c(antiAliasFactor_*antiAliasFactor_);
   for (int y = pixelsVertical_-1; y >= 0; y--) {
      for (uint32_t x = 0; x < pixelsHorizontal_; x++) {
         real_t r_sum = 0, g_sum = 0, b_sum = 0;
         for (uint32_t ay = uint32_c(y)*antiAliasFactor_; ay < (uint32_c(y+1))*antiAliasFactor_; ay++) {
            for (uint ax = x*antiAliasFactor_; ax < (x+1)*antiAliasFactor_; ax++) {
               size_t i = coordinateToArrayIndex(ax, ay);
               r_sum += real_c(intersectionsBuffer[i].r);
               g_sum += real_c(intersectionsBuffer[i].g);
               b_sum += real_c(intersectionsBuffer[i].b);
            }
         }
         u_char r = (u_char)(255 * (r_sum/patchSize));
         u_char g = (u_char)(255 * (g_sum/patchSize));
         u_char b = (u_char)(255 * (b_sum/patchSize));
         
         lodeImageBuffer[l] = r;
         lodeImageBuffer[l+1] = g;
         lodeImageBuffer[l+2] = b;
         l+=3;
      }
   }
   
   uint32_t error = lodepng::encode(fullPath.string(), lodeImageBuffer, getPixelsHorizontal(), getPixelsVertical(), LCT_RGB);
   if(error) {
      WALBERLA_LOG_WARNING("lodePNG error " << error << " when trying to save image file to " << fullPath.string() << ": " << lodepng_error_text(error));
   }
}


/*!\brief Conflate the intersectionsBuffer of each process onto the root process using MPI_Reduce.
 * \param intersectionsBuffer Buffer containing all intersections for entire image (including non-hits).
 * \param tt Optional TimingTree.
 *
 * This function conflates the intersectionsBuffer of each process onto the root process using the MPI_Reduce
 * routine. It requires sending intersection info structs for the entire image instead of only the ones of the hits.
 *
 * \attention This function only works on MPI builds due to the explicit usage of MPI routines.
 */
void Raytracer::syncImageUsingMPIReduce(std::vector<BodyIntersectionInfo>& intersectionsBuffer, WcTimingTree* tt) {
#ifdef WALBERLA_BUILD_WITH_MPI
   WALBERLA_MPI_BARRIER();
   if (tt != NULL) tt->start("Reduction");
   int rank = mpi::MPIManager::instance()->rank();

   const int recvRank = 0;
   if( rank == recvRank ) {
      MPI_Reduce(MPI_IN_PLACE,
                 &intersectionsBuffer[0], int_c(intersectionsBuffer.size()),
                 bodyIntersectionInfo_mpi_type, bodyIntersectionInfo_reduction_op,
                 recvRank, MPI_COMM_WORLD);
   } else {
      MPI_Reduce(&intersectionsBuffer[0], 0, int_c(intersectionsBuffer.size()),
                 bodyIntersectionInfo_mpi_type, bodyIntersectionInfo_reduction_op,
                 recvRank, MPI_COMM_WORLD);
   }
   
   WALBERLA_MPI_BARRIER();
   if (tt != NULL) tt->stop("Reduction");
#else
   WALBERLA_UNUSED(intersectionsBuffer);
   WALBERLA_UNUSED(tt);
   WALBERLA_ABORT("Cannot call MPI reduce on a non-MPI build due to usage of MPI-specific code.");
#endif
}
  
/*!\brief Conflate the intersectionsBuffer of each process onto the root process using MPI_Gather.
 * \param intersectionsBuffer Buffer containing intersections.
 * \param tt Optional TimingTree.
 *
 * This function conflates the intersectionsBuffer of each process onto the root process using the MPI_Gather
 * routine. It only sends information for hits.
 */
void Raytracer::syncImageUsingMPIGather(std::vector<BodyIntersectionInfo>& intersections, std::vector<BodyIntersectionInfo>& intersectionsBuffer, WcTimingTree* tt) {
   WALBERLA_MPI_BARRIER();
   if (tt != NULL) tt->start("Reduction");
   
   mpi::SendBuffer sendBuffer;
   for (auto& info: intersections) {
      sendBuffer << info.imageX << info.imageY
      << info.bodySystemID << info.t
      << info.r << info.g << info.b;
   }

   mpi::RecvBuffer recvBuffer;
   mpi::gathervBuffer(sendBuffer, recvBuffer, 0);
   
   WALBERLA_ROOT_SECTION() {
      BodyIntersectionInfo info;
      while (!recvBuffer.isEmpty()) {
         recvBuffer >> info.imageX;
         recvBuffer >> info.imageY;
         recvBuffer >> info.bodySystemID;
         recvBuffer >> info.t;
         recvBuffer >> info.r;
         recvBuffer >> info.g;
         recvBuffer >> info.b;
         
         size_t i = coordinateToArrayIndex(info.imageX, info.imageY);
         
         if (intersectionsBuffer[i].bodySystemID == 0 || info.t < intersectionsBuffer[i].t) {
            intersectionsBuffer[i] = info;
         }
      }
   }
   
   WALBERLA_MPI_BARRIER();
   if (tt != NULL) tt->stop("Reduction");
}

void Raytracer::localOutput(const std::vector<BodyIntersectionInfo>& intersectionsBuffer, size_t timestep, WcTimingTree* tt) {
   if (getImageOutputEnabled()) {
      if (getLocalImageOutputEnabled()) {
         if (tt != NULL) tt->start("Local Output");
         writeImageToFile(intersectionsBuffer, timestep);
         if (tt != NULL) tt->stop("Local Output");
      }
   }
   
   if (getTBufferOutputEnabled()) {
      if (tt != NULL) tt->start("Local Output");
      writeDepthsToFile(intersectionsBuffer, timestep);
      if (tt != NULL) tt->stop("Local Output");
   }
}

void Raytracer::output(const std::vector<BodyIntersectionInfo>& intersectionsBuffer, size_t timestep, WcTimingTree* tt) {
   WALBERLA_ROOT_SECTION() {
      if (tt != NULL) tt->start("Output");
      if (getImageOutputEnabled()) {
         writeImageToFile(intersectionsBuffer, timestep, true);
      }
      if (getTBufferOutputEnabled()) {
         writeDepthsToFile(intersectionsBuffer, timestep, true);
      }
      if (tt != NULL) tt->stop("Output");
   }
}

uint64_t Raytracer::naiveIntersectionTestCount = 0;

}
}
}
