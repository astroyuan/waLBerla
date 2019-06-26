#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from mesa_pd.accessor import Accessor
import mesa_pd.data as data
import mesa_pd.kernel as kernel
import mesa_pd.mpi as mpi

import argparse
import numpy as np
import os

if __name__ == '__main__':
   parser = argparse.ArgumentParser(description='Generate all necessary files for the waLBerla mesa_pd module.')
   parser.add_argument('path', help='Where should the files be created?')
   parser.add_argument("-f", "--force", help="Generate the files even if not inside a waLBerla directory.",
                       action="store_true")
   args = parser.parse_args()

   if ((not os.path.isfile(args.path + "/src/walberla.h")) and (not args.force)):
      raise RuntimeError(args.path + " is not the path to a waLBerla root directory! Specify -f to generate the files anyway.")

   os.makedirs(args.path + "/src/mesa_pd/common", exist_ok = True)
   os.makedirs(args.path + "/src/mesa_pd/data", exist_ok = True)
   os.makedirs(args.path + "/src/mesa_pd/domain", exist_ok = True)
   os.makedirs(args.path + "/src/mesa_pd/kernel", exist_ok = True)
   os.makedirs(args.path + "/src/mesa_pd/mpi/notifications", exist_ok = True)
   os.makedirs(args.path + "/src/mesa_pd/vtk", exist_ok = True)

   shapes = ["Sphere", "HalfSpace", "CylindricalBoundary", "Box"]

   ps    = data.ParticleStorage()
   ch    = data.ContactHistory()
   lc    = data.LinkedCells()
   ss    = data.ShapeStorage(ps, shapes)
   cs    = data.ContactStorage()

   ps.addProperty("position",         "walberla::mesa_pd::Vec3", defValue="real_t(0)", syncMode="ALWAYS")
   ps.addProperty("linearVelocity",   "walberla::mesa_pd::Vec3", defValue="real_t(0)", syncMode="ALWAYS")
   ps.addProperty("invMass",          "walberla::real_t",        defValue="real_t(1)", syncMode="COPY")
   ps.addProperty("force",            "walberla::mesa_pd::Vec3", defValue="real_t(0)", syncMode="NEVER")
   ps.addProperty("oldForce",         "walberla::mesa_pd::Vec3", defValue="real_t(0)", syncMode="MIGRATION")

   ps.addProperty("shapeID",          "size_t",                  defValue="",          syncMode="COPY")
   ps.addProperty("rotation",         "walberla::mesa_pd::Rot3", defValue="",          syncMode="ALWAYS")
   ps.addProperty("angularVelocity",  "walberla::mesa_pd::Vec3", defValue="real_t(0)", syncMode="ALWAYS")
   ps.addProperty("torque",           "walberla::mesa_pd::Vec3", defValue="real_t(0)", syncMode="NEVER")
   ps.addProperty("oldTorque",        "walberla::mesa_pd::Vec3", defValue="real_t(0)", syncMode="MIGRATION")

   ps.addProperty("type",             "uint_t",                  defValue="0",         syncMode="COPY")

   ps.addProperty("flags",            "walberla::mesa_pd::data::particle_flags::FlagT", defValue="", syncMode="COPY")
   ps.addProperty("nextParticle",     "int",                     defValue="-1",        syncMode="NEVER")

   ps.addProperty("oldContactHistory", "std::map<walberla::id_t, walberla::mesa_pd::data::ContactHistory>", defValue="", syncMode="ALWAYS")
   ps.addProperty("newContactHistory", "std::map<walberla::id_t, walberla::mesa_pd::data::ContactHistory>", defValue="", syncMode="NEVER")

   ps.addProperty("temperature",      "walberla::real_t",        defValue="real_t(0)", syncMode="ALWAYS")
   ps.addProperty("heatFlux",         "walberla::real_t",        defValue="real_t(0)", syncMode="NEVER")

   # Properties for HCSITS
   ps.addProperty("dv",               "walberla::mesa_pd::Vec3", defValue="real_t(0)", syncMode="NEVER")
   ps.addProperty("dw",               "walberla::mesa_pd::Vec3", defValue="real_t(0)", syncMode="NEVER")

   ch.addProperty("tangentialSpringDisplacement", "walberla::mesa_pd::Vec3", defValue="real_t(0)")
   ch.addProperty("isSticking",                   "bool",                    defValue="false")
   ch.addProperty("impactVelocityMagnitude",      "real_t",                  defValue="real_t(0)")

   cs.addProperty("id1",              "walberla::id_t",          defValue = "walberla::id_t(-1)", syncMode="NEVER")
   cs.addProperty("id2",              "walberla::id_t",          defValue = "walberla::id_t(-1)", syncMode="NEVER")
   cs.addProperty("distance",         "real_t",                  defValue = "real_t(1)",          syncMode="NEVER")
   cs.addProperty("normal",           "walberla::mesa_pd::Vec3", defValue = "real_t(0)",          syncMode="NEVER")
   cs.addProperty("position",         "walberla::mesa_pd::Vec3", defValue = "real_t(0)",          syncMode="NEVER")
   cs.addProperty("t",                "walberla::mesa_pd::Vec3", defValue = "real_t(0)",          syncMode="NEVER")
   cs.addProperty("o",                "walberla::mesa_pd::Vec3", defValue = "real_t(0)",          syncMode="NEVER")
   cs.addProperty("r1",               "walberla::mesa_pd::Vec3", defValue = "real_t(0)",          syncMode="NEVER")
   cs.addProperty("r2",               "walberla::mesa_pd::Vec3", defValue = "real_t(0)",          syncMode="NEVER")
   cs.addProperty("mu",               "real_t",                  defValue = "real_t(0)",          syncMode="NEVER")
   cs.addProperty("p",                "walberla::mesa_pd::Vec3", defValue = "real_t(0)",          syncMode="NEVER")
   cs.addProperty("diag_nto",         "walberla::mesa_pd::Mat3", defValue = "real_t(0)",          syncMode="NEVER")
   cs.addProperty("diag_nto_inv",     "walberla::mesa_pd::Mat3", defValue = "real_t(0)",          syncMode="NEVER")
   cs.addProperty("diag_to_inv",      "walberla::mesa_pd::Mat2", defValue = "real_t(0)",          syncMode="NEVER")
   cs.addProperty("diag_n_inv",       "real_t",                  defValue = "real_t(0)",          syncMode="NEVER")
   cs.addProperty("p",                "walberla::mesa_pd::Vec3", defValue = "real_t(0)",          syncMode="NEVER")


   kernels = []
   kernels.append( kernel.DetectAndStoreContacts() )
   kernels.append( kernel.DoubleCast(shapes) )
   kernels.append( kernel.ExplicitEuler() )
   kernels.append( kernel.ExplicitEulerWithShape() )
   kernels.append( kernel.ForceLJ() )
   kernels.append( kernel.HeatConduction() )
   kernels.append( kernel.InsertParticleIntoLinkedCells() )
   kernels.append( kernel.LinearSpringDashpot() )
   kernels.append( kernel.NonLinearSpringDashpot() )
   kernels.append( kernel.SingleCast(shapes) )
   kernels.append( kernel.SpringDashpot() )
   kernels.append( kernel.TemperatureIntegration() )
   kernels.append( kernel.VelocityVerlet() )
   kernels.append( kernel.VelocityVerletWithShape() )


   ac = Accessor()
   for k in kernels:
      ac.mergeRequirements(k.getRequirements())
   ac.printSummary()

   comm = []
   comm.append(mpi.BroadcastProperty())
   comm.append(mpi.ClearNextNeighborSync())
   comm.append(mpi.ReduceContactHistory())
   comm.append(mpi.ReduceProperty())
   comm.append(mpi.SyncGhostOwners(ps))
   comm.append(mpi.SyncNextNeighbors(ps))


   ps.generate(args.path + "/src/mesa_pd/")
   ch.generate(args.path + "/src/mesa_pd/")
   lc.generate(args.path + "/src/mesa_pd/")
   ss.generate(args.path + "/src/mesa_pd/")
   cs.generate(args.path + "/src/mesa_pd/")

   for k in kernels:
      k.generate(args.path + "/src/mesa_pd/")

   for c in comm:
      c.generate(args.path + "/src/mesa_pd/")
