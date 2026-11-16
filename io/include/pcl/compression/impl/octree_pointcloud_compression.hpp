/*
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2009-2012, Willow Garage, Inc.
 * 
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 */

#ifndef OCTREE_COMPRESSION_HPP
#define OCTREE_COMPRESSION_HPP

#include <pcl/common/io.h> // for getFieldIndex
#include <pcl/compression/entropy_range_coder.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>
#include <cstring>

namespace pcl
{
  namespace io
  {
    //////////////////////////////////////////////////////////////////////////////////////////////
    template<typename PointT, typename LeafT, typename BranchT, typename OctreeT> void OctreePointCloudCompression<
        PointT, LeafT, BranchT, OctreeT>::encodePointCloud (
        const PointCloudConstPtr &cloud_arg,
        std::ostream& compressed_tree_data_out_arg)
    {
      auto recent_tree_depth =
          static_cast<unsigned char> (this->getTreeDepth ());

      // initialize octree
      this->setInputCloud (cloud_arg);

      // add point to octree
      this->addPointsFromInputCloud ();

      // make sure cloud contains points
      if (this->leaf_count_>0) {

        // analyze available point attributes and apply encoding configuration
        analyzePointFields ();
        cloud_with_color_ &= do_color_encoding_;
        cloud_with_normals_ &= do_normal_encoding_;

        // lossless coding requires the original point attributes, so no voxel can be dropped
        if (lossless_)
          do_voxel_grid_enDecoding_ = false;

        data_with_color_ = cloud_with_color_;
        data_with_normals_ = cloud_with_normals_;
        data_lossless_ = lossless_;
        v2_frame_ = data_lossless_ || data_with_normals_;

        source_width_ = cloud_arg->width;
        source_height_ = cloud_arg->height;
        source_is_dense_ = cloud_arg->is_dense;


        // if octree depth changed, we enforce I-frame encoding
        i_frame_ |= (recent_tree_depth != this->getTreeDepth ());// | !(iFrameCounter%10);

        // enable I-frame rate
        if (i_frame_counter_++==i_frame_rate_)
        {
          i_frame_counter_ =0;
          i_frame_ = true;
        }

        // increase frameID
        frame_ID_++;

        // do octree encoding
        if (!do_voxel_grid_enDecoding_)
        {
          point_count_data_vector_.clear ();
          point_count_data_vector_.reserve (cloud_arg->size ());
        }
        point_index_data_vector_.clear ();
        lossless_point_data_vector_.clear ();
        normal_data_vector_.clear ();
        if (data_lossless_)
        {
          point_count_data_vector_.clear ();
          point_count_data_vector_.reserve (cloud_arg->size ());
          point_index_data_vector_.reserve (cloud_arg->size ());
          lossless_point_data_vector_.reserve (cloud_arg->size () * lossless_point_record_size_);
        }
        else if (data_with_normals_)
        {
          normal_data_vector_.reserve (cloud_arg->size () * 3 * sizeof (float));
        }

        // initialize color encoding
        color_coder_.initializeEncoding ();
        color_coder_.setPointCount (static_cast<unsigned int> (cloud_arg->size ()));
        color_coder_.setVoxelCount (static_cast<unsigned int> (this->leaf_count_));

        // initialize point encoding
        point_coder_.initializeEncoding ();
        point_coder_.setPointCount (static_cast<unsigned int> (cloud_arg->size ()));

        // serialize octree
        if (i_frame_)
          // i-frame encoding - encode tree structure without referencing previous buffer
          this->serializeTree (binary_tree_data_vector_, false);
        else
          // p-frame encoding - XOR encoded tree structure
          this->serializeTree (binary_tree_data_vector_, true);


        // write frame header information to stream
        this->writeFrameHeader (compressed_tree_data_out_arg);

        // apply entropy coding to the content of all data vectors and send data to output stream
        this->entropyEncoding (compressed_tree_data_out_arg);

        // prepare for next frame
        this->switchBuffers ();

        // reset object count
        object_count_ = 0;

        if (b_show_statistics_)
        {
          float bytes_per_XYZ = static_cast<float> (compressed_point_data_len_) / static_cast<float> (point_count_);
          float bytes_per_color = static_cast<float> (compressed_color_data_len_) / static_cast<float> (point_count_);
          float bytes_per_normal = static_cast<float> (compressed_normal_data_len_) / static_cast<float> (point_count_);

          PCL_INFO ("*** POINTCLOUD ENCODING ***\n");
          PCL_INFO ("Frame ID: %d\n", frame_ID_);
          if (i_frame_)
            PCL_INFO ("Encoding Frame: Intra frame\n");
          else
            PCL_INFO ("Encoding Frame: Prediction frame\n");
          PCL_INFO ("Number of encoded points: %ld\n", point_count_);
          PCL_INFO ("XYZ compression percentage: %f%%\n", bytes_per_XYZ / (3.0f * sizeof (float)) * 100.0f);
          PCL_INFO ("XYZ bytes per point: %f bytes\n", bytes_per_XYZ);
          PCL_INFO ("Color compression percentage: %f%%\n", bytes_per_color / (sizeof (int)) * 100.0f);
          PCL_INFO ("Color bytes per point: %f bytes\n", bytes_per_color);
          if (data_with_normals_)
          {
            PCL_INFO ("Normal compression percentage: %f%%\n", bytes_per_normal / (3.0f * sizeof (float)) * 100.0f);
            PCL_INFO ("Normal bytes per point: %f bytes\n", bytes_per_normal);
          }
          PCL_INFO ("Size of uncompressed point cloud: %f kBytes\n", static_cast<float> (point_count_) * (sizeof (int) + 3.0f * sizeof (float)) / 1024.0f);
          PCL_INFO ("Size of compressed point cloud: %f kBytes\n", static_cast<float> (compressed_point_data_len_ + compressed_color_data_len_ + compressed_normal_data_len_) / 1024.0f);
          PCL_INFO ("Total bytes per point: %f bytes\n", bytes_per_XYZ + bytes_per_color + bytes_per_normal);
          PCL_INFO ("Total compression percentage: %f%%\n", (bytes_per_XYZ + bytes_per_color + bytes_per_normal) / (sizeof (int) + 3.0f * sizeof (float)) * 100.0f);
          PCL_INFO ("Compression ratio: %f\n\n", static_cast<float> (sizeof (int) + 3.0f * sizeof (float)) / static_cast<float> (bytes_per_XYZ + bytes_per_color + bytes_per_normal));
        }
        
        i_frame_ = false;
      } else {
        if (b_show_statistics_)
        PCL_INFO ("Info: Dropping empty point cloud\n");
        this->deleteTree();
        i_frame_counter_ = 0;
        i_frame_ = true;
      }
    }

    //////////////////////////////////////////////////////////////////////////////////////////////
    template<typename PointT, typename LeafT, typename BranchT, typename OctreeT> void
    OctreePointCloudCompression<PointT, LeafT, BranchT, OctreeT>::decodePointCloud (
        std::istream& compressed_tree_data_in_arg,
        PointCloudPtr &cloud_arg)
    {

      // synchronize to frame header
      if (!syncToHeader(compressed_tree_data_in_arg))
        return;

      // initialize octree
      this->switchBuffers ();
      this->setOutputCloud (cloud_arg);

      // analyze the point fields available to the decoder
      analyzePointFields ();

      // read header from input stream
      this->readFrameHeader (compressed_tree_data_in_arg);

      // decode data vectors from stream
      this->entropyDecoding (compressed_tree_data_in_arg);

      // initialize color and point encoding
      color_coder_.initializeDecoding ();
      point_coder_.initializeDecoding ();

      // initialize output cloud
      output_->points.clear ();
      output_->points.reserve (static_cast<std::size_t> (point_count_));

      if (i_frame_)
        // i-frame decoding - decode tree structure without referencing previous buffer
        this->deserializeTree (binary_tree_data_vector_, false);
      else
        // p-frame decoding - decode XOR encoded tree structure
        this->deserializeTree (binary_tree_data_vector_, true);

      // assign point cloud properties
      if (data_lossless_)
      {
        restoreOriginalPointOrder ();
        output_->width = source_width_;
        output_->height = source_height_;
        output_->is_dense = source_is_dense_;
      }
      else
      {
        output_->height = 1;
        output_->width = cloud_arg->size ();
        output_->is_dense = false;
      }

      if (b_show_statistics_)
      {
        float bytes_per_XYZ = static_cast<float> (compressed_point_data_len_) / static_cast<float> (point_count_);
        float bytes_per_color = static_cast<float> (compressed_color_data_len_) / static_cast<float> (point_count_);
        float bytes_per_normal = static_cast<float> (compressed_normal_data_len_) / static_cast<float> (point_count_);

        PCL_INFO ("*** POINTCLOUD DECODING ***\n");
        PCL_INFO ("Frame ID: %d\n", frame_ID_);
        if (i_frame_)
          PCL_INFO ("Decoding Frame: Intra frame\n");
        else
          PCL_INFO ("Decoding Frame: Prediction frame\n");
        PCL_INFO ("Number of decoded points: %ld\n", point_count_);
        PCL_INFO ("XYZ compression percentage: %f%%\n", bytes_per_XYZ / (3.0f * sizeof (float)) * 100.0f);
        PCL_INFO ("XYZ bytes per point: %f bytes\n", bytes_per_XYZ);
        PCL_INFO ("Color compression percentage: %f%%\n", bytes_per_color / (sizeof (int)) * 100.0f);
        PCL_INFO ("Color bytes per point: %f bytes\n", bytes_per_color);
        if (data_with_normals_)
        {
          PCL_INFO ("Normal compression percentage: %f%%\n", bytes_per_normal / (3.0f * sizeof (float)) * 100.0f);
          PCL_INFO ("Normal bytes per point: %f bytes\n", bytes_per_normal);
        }
        PCL_INFO ("Size of uncompressed point cloud: %f kBytes\n", static_cast<float> (point_count_) * (sizeof (int) + 3.0f * sizeof (float)) / 1024.0f);
        PCL_INFO ("Size of compressed point cloud: %f kBytes\n", static_cast<float> (compressed_point_data_len_ + compressed_color_data_len_ + compressed_normal_data_len_) / 1024.0f);
        PCL_INFO ("Total bytes per point: %f bytes\n", bytes_per_XYZ + bytes_per_color + bytes_per_normal);
        PCL_INFO ("Total compression percentage: %f%%\n", (bytes_per_XYZ + bytes_per_color + bytes_per_normal) / (sizeof (int) + 3.0f * sizeof (float)) * 100.0f);
        PCL_INFO ("Compression ratio: %f\n\n", static_cast<float> (sizeof (int) + 3.0f * sizeof (float)) / static_cast<float> (bytes_per_XYZ + bytes_per_color + bytes_per_normal));
      }
    }

    //////////////////////////////////////////////////////////////////////////////////////////////
    template<typename PointT, typename LeafT, typename BranchT, typename OctreeT> void
    OctreePointCloudCompression<PointT, LeafT, BranchT, OctreeT>::entropyEncoding (std::ostream& compressed_tree_data_out_arg)
    {
      std::uint64_t binary_tree_data_vector_size;
      std::uint64_t point_avg_color_data_vector_size;

      compressed_point_data_len_ = 0;
      compressed_color_data_len_ = 0;
      compressed_normal_data_len_ = 0;

      // encode binary octree structure
      binary_tree_data_vector_size = binary_tree_data_vector_.size ();
      compressed_tree_data_out_arg.write (reinterpret_cast<const char*> (&binary_tree_data_vector_size), sizeof (binary_tree_data_vector_size));
      compressed_point_data_len_ += entropy_coder_.encodeCharVectorToStream (binary_tree_data_vector_,
                                                                             compressed_tree_data_out_arg);

      if (cloud_with_color_)
      {
        // encode averaged voxel color information
        std::vector<char>& pointAvgColorDataVector = color_coder_.getAverageDataVector ();
        point_avg_color_data_vector_size = pointAvgColorDataVector.size ();
        compressed_tree_data_out_arg.write (reinterpret_cast<const char*> (&point_avg_color_data_vector_size),
                                            sizeof (point_avg_color_data_vector_size));
        compressed_color_data_len_ += entropy_coder_.encodeCharVectorToStream (pointAvgColorDataVector,
                                                                               compressed_tree_data_out_arg);
      }

      if (data_with_normals_ && !data_lossless_)
      {
        std::uint64_t normal_data_vector_size = normal_data_vector_.size ();
        compressed_tree_data_out_arg.write (reinterpret_cast<const char*> (&normal_data_vector_size),
                                            sizeof (normal_data_vector_size));
        compressed_normal_data_len_ += entropy_coder_.encodeCharVectorToStream (normal_data_vector_,
                                                                                 compressed_tree_data_out_arg);
      }

      if (data_lossless_)
      {
        std::uint64_t point_count_data_vector_size = point_count_data_vector_.size ();
        compressed_tree_data_out_arg.write (reinterpret_cast<const char*> (&point_count_data_vector_size),
                                            sizeof (point_count_data_vector_size));
        compressed_point_data_len_ += entropy_coder_.encodeIntVectorToStream (point_count_data_vector_,
                                                                              compressed_tree_data_out_arg);

        std::uint64_t lossless_point_data_vector_size = lossless_point_data_vector_.size ();
        compressed_tree_data_out_arg.write (reinterpret_cast<const char*> (&lossless_point_data_vector_size),
                                            sizeof (lossless_point_data_vector_size));
        compressed_point_data_len_ += entropy_coder_.encodeCharVectorToStream (lossless_point_data_vector_,
                                                                               compressed_tree_data_out_arg);

        std::uint64_t point_index_data_vector_size = point_index_data_vector_.size ();
        compressed_tree_data_out_arg.write (reinterpret_cast<const char*> (&point_index_data_vector_size),
                                            sizeof (point_index_data_vector_size));
        compressed_point_data_len_ += entropy_coder_.encodeIntVectorToStream (point_index_data_vector_,
                                                                              compressed_tree_data_out_arg);
      }
      else if (!do_voxel_grid_enDecoding_)
      {
        std::uint64_t pointCountDataVector_size;
        std::uint64_t point_diff_data_vector_size;
        std::uint64_t point_diff_color_data_vector_size;

        // encode amount of points per voxel
        pointCountDataVector_size = point_count_data_vector_.size ();
        compressed_tree_data_out_arg.write (reinterpret_cast<const char*> (&pointCountDataVector_size), sizeof (pointCountDataVector_size));
        compressed_point_data_len_ += entropy_coder_.encodeIntVectorToStream (point_count_data_vector_,
                                                                          compressed_tree_data_out_arg);

        // encode differential point information
        std::vector<char>& point_diff_data_vector = point_coder_.getDifferentialDataVector ();
        point_diff_data_vector_size = point_diff_data_vector.size ();
        compressed_tree_data_out_arg.write (reinterpret_cast<const char*> (&point_diff_data_vector_size), sizeof (point_diff_data_vector_size));
        compressed_point_data_len_ += entropy_coder_.encodeCharVectorToStream (point_diff_data_vector,
                                                                               compressed_tree_data_out_arg);
        if (cloud_with_color_)
        {
          // encode differential color information
          std::vector<char>& point_diff_color_data_vector = color_coder_.getDifferentialDataVector ();
          point_diff_color_data_vector_size = point_diff_color_data_vector.size ();
          compressed_tree_data_out_arg.write (reinterpret_cast<const char*> (&point_diff_color_data_vector_size),
                                           sizeof (point_diff_color_data_vector_size));
          compressed_color_data_len_ += entropy_coder_.encodeCharVectorToStream (point_diff_color_data_vector,
                                                                                 compressed_tree_data_out_arg);
        }
      }
      // flush output stream
      compressed_tree_data_out_arg.flush ();
    }

    //////////////////////////////////////////////////////////////////////////////////////////////
    template<typename PointT, typename LeafT, typename BranchT, typename OctreeT> void
    OctreePointCloudCompression<PointT, LeafT, BranchT, OctreeT>::entropyDecoding (std::istream& compressed_tree_data_in_arg)
    {
      std::uint64_t binary_tree_data_vector_size;
      std::uint64_t point_avg_color_data_vector_size;

      compressed_point_data_len_ = 0;
      compressed_color_data_len_ = 0;
      compressed_normal_data_len_ = 0;
      normal_data_position_ = 0;
      lossless_point_data_position_ = 0;

      // decode binary octree structure
      compressed_tree_data_in_arg.read (reinterpret_cast<char*> (&binary_tree_data_vector_size), sizeof (binary_tree_data_vector_size));
      binary_tree_data_vector_.resize (static_cast<std::size_t> (binary_tree_data_vector_size));
      compressed_point_data_len_ += entropy_coder_.decodeStreamToCharVector (compressed_tree_data_in_arg,
                                                                         binary_tree_data_vector_);

      if (data_with_color_)
      {
        // decode averaged voxel color information
        std::vector<char>& point_avg_color_data_vector = color_coder_.getAverageDataVector ();
        compressed_tree_data_in_arg.read (reinterpret_cast<char*> (&point_avg_color_data_vector_size), sizeof (point_avg_color_data_vector_size));
        point_avg_color_data_vector.resize (static_cast<std::size_t> (point_avg_color_data_vector_size));
        compressed_color_data_len_ += entropy_coder_.decodeStreamToCharVector (compressed_tree_data_in_arg,
                                                                           point_avg_color_data_vector);
      }

      if (data_with_normals_ && !data_lossless_)
      {
        std::uint64_t normal_data_vector_size;
        compressed_tree_data_in_arg.read (reinterpret_cast<char*> (&normal_data_vector_size),
                                          sizeof (normal_data_vector_size));
        normal_data_vector_.resize (static_cast<std::size_t> (normal_data_vector_size));
        compressed_normal_data_len_ += entropy_coder_.decodeStreamToCharVector (compressed_tree_data_in_arg,
                                                                                normal_data_vector_);
      }
      else
      {
        normal_data_vector_.clear ();
      }

      if (data_lossless_)
      {
        std::uint64_t point_count_data_vector_size;
        compressed_tree_data_in_arg.read (reinterpret_cast<char*> (&point_count_data_vector_size),
                                          sizeof (point_count_data_vector_size));
        point_count_data_vector_.resize (static_cast<std::size_t> (point_count_data_vector_size));
        compressed_point_data_len_ += entropy_coder_.decodeStreamToIntVector (compressed_tree_data_in_arg,
                                                                               point_count_data_vector_);
        point_count_data_vector_iterator_ = point_count_data_vector_.begin ();

        std::uint64_t lossless_point_data_vector_size;
        compressed_tree_data_in_arg.read (reinterpret_cast<char*> (&lossless_point_data_vector_size),
                                          sizeof (lossless_point_data_vector_size));
        lossless_point_data_vector_.resize (static_cast<std::size_t> (lossless_point_data_vector_size));
        compressed_point_data_len_ += entropy_coder_.decodeStreamToCharVector (compressed_tree_data_in_arg,
                                                                                lossless_point_data_vector_);

        std::uint64_t point_index_data_vector_size;
        compressed_tree_data_in_arg.read (reinterpret_cast<char*> (&point_index_data_vector_size),
                                          sizeof (point_index_data_vector_size));
        point_index_data_vector_.resize (static_cast<std::size_t> (point_index_data_vector_size));
        compressed_point_data_len_ += entropy_coder_.decodeStreamToIntVector (compressed_tree_data_in_arg,
                                                                               point_index_data_vector_);
      }
      else if (!do_voxel_grid_enDecoding_)
      {
        std::uint64_t point_count_data_vector_size;
        std::uint64_t point_diff_data_vector_size;
        std::uint64_t point_diff_color_data_vector_size;

        // decode amount of points per voxel
        compressed_tree_data_in_arg.read (reinterpret_cast<char*> (&point_count_data_vector_size), sizeof (point_count_data_vector_size));
        point_count_data_vector_.resize (static_cast<std::size_t> (point_count_data_vector_size));
        compressed_point_data_len_ += entropy_coder_.decodeStreamToIntVector (compressed_tree_data_in_arg, point_count_data_vector_);
        point_count_data_vector_iterator_ = point_count_data_vector_.begin ();

        // decode differential point information
        std::vector<char>& pointDiffDataVector = point_coder_.getDifferentialDataVector ();
        compressed_tree_data_in_arg.read (reinterpret_cast<char*> (&point_diff_data_vector_size), sizeof (point_diff_data_vector_size));
        pointDiffDataVector.resize (static_cast<std::size_t> (point_diff_data_vector_size));
        compressed_point_data_len_ += entropy_coder_.decodeStreamToCharVector (compressed_tree_data_in_arg,
                                                                           pointDiffDataVector);

        if (data_with_color_)
        {
          // decode differential color information
          std::vector<char>& pointDiffColorDataVector = color_coder_.getDifferentialDataVector ();
          compressed_tree_data_in_arg.read (reinterpret_cast<char*> (&point_diff_color_data_vector_size), sizeof (point_diff_color_data_vector_size));
          pointDiffColorDataVector.resize (static_cast<std::size_t> (point_diff_color_data_vector_size));
          compressed_color_data_len_ += entropy_coder_.decodeStreamToCharVector (compressed_tree_data_in_arg,
                                                                             pointDiffColorDataVector);
        }
      }
    }

    //////////////////////////////////////////////////////////////////////////////////////////////
    template<typename PointT, typename LeafT, typename BranchT, typename OctreeT> void
    OctreePointCloudCompression<PointT, LeafT, BranchT, OctreeT>::writeFrameHeader (std::ostream& compressed_tree_data_out_arg)
    {
      const char* frame_identifier = v2_frame_ ? frame_header_identifier_v2_ : frame_header_identifier_;

      // encode header identifier
      compressed_tree_data_out_arg.write (reinterpret_cast<const char*> (frame_identifier), strlen (frame_identifier));
      // encode point cloud header id
      compressed_tree_data_out_arg.write (reinterpret_cast<const char*> (&frame_ID_), sizeof (frame_ID_));
      // encode frame type (I/P-frame)
      compressed_tree_data_out_arg.write (reinterpret_cast<const char*> (&i_frame_), sizeof (i_frame_));
      if (i_frame_)
      {
        double min_x, min_y, min_z, max_x, max_y, max_z;
        double octree_resolution;
        unsigned char color_bit_depth;
        double point_resolution;

        // get current configuration
        octree_resolution = this->getResolution ();
        color_bit_depth  = color_coder_.getBitDepth ();
        point_resolution= point_coder_.getPrecision ();
        this->getBoundingBox (min_x, min_y, min_z, max_x, max_y, max_z);

        // encode amount of points
        if (do_voxel_grid_enDecoding_)
          point_count_ = this->leaf_count_;
        else
          point_count_ = this->object_count_;

        // encode coding configuration
        compressed_tree_data_out_arg.write (reinterpret_cast<const char*> (&do_voxel_grid_enDecoding_), sizeof (do_voxel_grid_enDecoding_));
        if (v2_frame_)
        {
          compressed_tree_data_out_arg.write (reinterpret_cast<const char*> (&data_lossless_), sizeof (data_lossless_));
          compressed_tree_data_out_arg.write (reinterpret_cast<const char*> (&data_with_color_), sizeof (data_with_color_));
          compressed_tree_data_out_arg.write (reinterpret_cast<const char*> (&data_with_normals_), sizeof (data_with_normals_));
        }
        else
        {
          compressed_tree_data_out_arg.write (reinterpret_cast<const char*> (&cloud_with_color_), sizeof (cloud_with_color_));
        }
        compressed_tree_data_out_arg.write (reinterpret_cast<const char*> (&point_count_), sizeof (point_count_));
        compressed_tree_data_out_arg.write (reinterpret_cast<const char*> (&octree_resolution), sizeof (octree_resolution));
        compressed_tree_data_out_arg.write (reinterpret_cast<const char*> (&color_bit_depth), sizeof (color_bit_depth));
        compressed_tree_data_out_arg.write (reinterpret_cast<const char*> (&point_resolution), sizeof (point_resolution));
        if (v2_frame_)
        {
          compressed_tree_data_out_arg.write (reinterpret_cast<const char*> (&source_width_), sizeof (source_width_));
          compressed_tree_data_out_arg.write (reinterpret_cast<const char*> (&source_height_), sizeof (source_height_));
          compressed_tree_data_out_arg.write (reinterpret_cast<const char*> (&source_is_dense_), sizeof (source_is_dense_));
          const auto record_size = static_cast<std::uint64_t> (lossless_point_record_size_);
          compressed_tree_data_out_arg.write (reinterpret_cast<const char*> (&record_size), sizeof (record_size));
        }

        // encode octree bounding box
        compressed_tree_data_out_arg.write (reinterpret_cast<const char*> (&min_x), sizeof (min_x));
        compressed_tree_data_out_arg.write (reinterpret_cast<const char*> (&min_y), sizeof (min_y));
        compressed_tree_data_out_arg.write (reinterpret_cast<const char*> (&min_z), sizeof (min_z));
        compressed_tree_data_out_arg.write (reinterpret_cast<const char*> (&max_x), sizeof (max_x));
        compressed_tree_data_out_arg.write (reinterpret_cast<const char*> (&max_y), sizeof (max_y));
        compressed_tree_data_out_arg.write (reinterpret_cast<const char*> (&max_z), sizeof (max_z));
      }
    }

    //////////////////////////////////////////////////////////////////////////////////////////////
    template<typename PointT, typename LeafT, typename BranchT, typename OctreeT> bool
    OctreePointCloudCompression<PointT, LeafT, BranchT, OctreeT>::syncToHeader ( std::istream& compressed_tree_data_in_arg)
    {
      const std::string v1_identifier (frame_header_identifier_);
      const std::string v2_identifier (frame_header_identifier_v2_);
      const std::size_t max_identifier_size = std::max (v1_identifier.size (), v2_identifier.size ());

      std::string window;
      window.reserve (max_identifier_size);
      char read_char;
      while (compressed_tree_data_in_arg.get (read_char))
      {
        window.push_back (read_char);
        if (window.size () > max_identifier_size)
          window.erase (window.begin ());

        if (window == v2_identifier)
        {
          v2_frame_ = true;
          return true;
        }
        if (window == v1_identifier)
        {
          v2_frame_ = false;
          return true;
        }
      }

      PCL_ERROR ("[pcl::io::OctreePointCloudCompression::syncToHeader] Could not find a frame header.\n");
      return false;
    }

    //////////////////////////////////////////////////////////////////////////////////////////////
    template<typename PointT, typename LeafT, typename BranchT, typename OctreeT> void
    OctreePointCloudCompression<PointT, LeafT, BranchT, OctreeT>::readFrameHeader ( std::istream& compressed_tree_data_in_arg)
    {
      // read header
      compressed_tree_data_in_arg.read (reinterpret_cast<char*> (&frame_ID_), sizeof (frame_ID_));
      compressed_tree_data_in_arg.read (reinterpret_cast<char*>(&i_frame_), sizeof (i_frame_));
      if (i_frame_)
      {
        double min_x, min_y, min_z, max_x, max_y, max_z;
        double octree_resolution;
        unsigned char color_bit_depth;
        double point_resolution;

        // read coder configuration
        compressed_tree_data_in_arg.read (reinterpret_cast<char*> (&do_voxel_grid_enDecoding_), sizeof (do_voxel_grid_enDecoding_));
        if (v2_frame_)
        {
          compressed_tree_data_in_arg.read (reinterpret_cast<char*> (&data_lossless_), sizeof (data_lossless_));
          compressed_tree_data_in_arg.read (reinterpret_cast<char*> (&data_with_color_), sizeof (data_with_color_));
          compressed_tree_data_in_arg.read (reinterpret_cast<char*> (&data_with_normals_), sizeof (data_with_normals_));
        }
        else
        {
          compressed_tree_data_in_arg.read (reinterpret_cast<char*> (&data_with_color_), sizeof (data_with_color_));
          data_with_normals_ = false;
          data_lossless_ = false;
        }
        compressed_tree_data_in_arg.read (reinterpret_cast<char*> (&point_count_), sizeof (point_count_));
        compressed_tree_data_in_arg.read (reinterpret_cast<char*> (&octree_resolution), sizeof (octree_resolution));
        compressed_tree_data_in_arg.read (reinterpret_cast<char*> (&color_bit_depth), sizeof (color_bit_depth));
        compressed_tree_data_in_arg.read (reinterpret_cast<char*> (&point_resolution), sizeof (point_resolution));
        if (v2_frame_)
        {
          compressed_tree_data_in_arg.read (reinterpret_cast<char*> (&source_width_), sizeof (source_width_));
          compressed_tree_data_in_arg.read (reinterpret_cast<char*> (&source_height_), sizeof (source_height_));
          compressed_tree_data_in_arg.read (reinterpret_cast<char*> (&source_is_dense_), sizeof (source_is_dense_));

          std::uint64_t encoded_record_size;
          compressed_tree_data_in_arg.read (reinterpret_cast<char*> (&encoded_record_size), sizeof (encoded_record_size));
          if (data_lossless_ && encoded_record_size != lossless_point_record_size_)
          {
            PCL_WARN ("[pcl::io::OctreePointCloudCompression::readFrameHeader] Lossless point record size mismatch (stream: %zu, decoder: %zu).\n",
                      static_cast<std::size_t> (encoded_record_size), lossless_point_record_size_);
          }
        }

        // read octree bounding box
        compressed_tree_data_in_arg.read (reinterpret_cast<char*> (&min_x), sizeof (min_x));
        compressed_tree_data_in_arg.read (reinterpret_cast<char*> (&min_y), sizeof (min_y));
        compressed_tree_data_in_arg.read (reinterpret_cast<char*> (&min_z), sizeof (min_z));
        compressed_tree_data_in_arg.read (reinterpret_cast<char*> (&max_x), sizeof (max_x));
        compressed_tree_data_in_arg.read (reinterpret_cast<char*> (&max_y), sizeof (max_y));
        compressed_tree_data_in_arg.read (reinterpret_cast<char*> (&max_z), sizeof (max_z));

        // reset octree and assign new bounding box & resolution
        this->deleteTree ();
        this->setResolution (octree_resolution);
        this->defineBoundingBox (min_x, min_y, min_z, max_x, max_y, max_z);

        // configure color & point coding
        color_coder_.setBitDepth (color_bit_depth);
        point_coder_.setPrecision (static_cast<float> (point_resolution));
      }
    }

    //////////////////////////////////////////////////////////////////////////////////////////////
    template<typename PointT, typename LeafT, typename BranchT, typename OctreeT> void
    OctreePointCloudCompression<PointT, LeafT, BranchT, OctreeT>::serializeTreeCallback (
        LeafT &leaf_arg, const OctreeKey & key_arg)
    {
      // reference to point indices vector stored within octree leaf
      const auto& leafIdx = leaf_arg.getPointIndicesVector();

      if (data_lossless_)
      {
        point_count_data_vector_.push_back (static_cast<unsigned int> (leafIdx.size ()));
        for (const auto& idx : leafIdx)
        {
          point_index_data_vector_.push_back (static_cast<unsigned int> (idx));
          encodeLosslessPoint ((*this->input_)[idx]);
        }
        return;
      }

      if (!do_voxel_grid_enDecoding_)
      {
        double lowerVoxelCorner[3];

        // encode amount of points within voxel
        point_count_data_vector_.push_back (static_cast<int> (leafIdx.size ()));

        // calculate lower voxel corner based on octree key
        lowerVoxelCorner[0] = static_cast<double> (key_arg.x) * this->resolution_ + this->min_x_;
        lowerVoxelCorner[1] = static_cast<double> (key_arg.y) * this->resolution_ + this->min_y_;
        lowerVoxelCorner[2] = static_cast<double> (key_arg.z) * this->resolution_ + this->min_z_;

        // differentially encode points to lower voxel corner
        point_coder_.encodePoints (leafIdx, lowerVoxelCorner, this->input_);

        if (cloud_with_color_)
          // encode color of points
          color_coder_.encodePoints (leafIdx, point_color_offset_, this->input_);

        if (cloud_with_normals_)
        {
          for (const auto& idx : leafIdx)
            encodeNormal ((*this->input_)[idx]);
        }
      }
      else
      {
        if (cloud_with_color_)
          // encode average color of all points within voxel
          color_coder_.encodeAverageOfPoints (leafIdx, point_color_offset_, this->input_);

        if (cloud_with_normals_)
        {
          float average_normal[3] = {0.0f, 0.0f, 0.0f};
          for (const auto& idx : leafIdx)
          {
            const char* point_data = reinterpret_cast<const char*> (&(*this->input_)[idx]);
            for (int component = 0; component < 3; ++component)
            {
              float value;
              std::memcpy (&value, point_data + normal_offsets_[component], sizeof (value));
              average_normal[component] += value;
            }
          }

          const float normal_length = std::sqrt (average_normal[0] * average_normal[0] +
                                                 average_normal[1] * average_normal[1] +
                                                 average_normal[2] * average_normal[2]);
          if (normal_length > 1e-12f)
          {
            for (float& value : average_normal)
              value /= normal_length;
          }
          normal_data_vector_.insert (normal_data_vector_.end (),
                                      reinterpret_cast<const char*> (average_normal),
                                      reinterpret_cast<const char*> (average_normal) + sizeof (average_normal));
        }
      }
    }

    //////////////////////////////////////////////////////////////////////////////////////////////
    template<typename PointT, typename LeafT, typename BranchT, typename OctreeT> void
    OctreePointCloudCompression<PointT, LeafT, BranchT, OctreeT>::deserializeTreeCallback (LeafT&,
        const OctreeKey& key_arg)
    {
      PointT newPoint;

      std::size_t pointCount = 1;

      if (data_lossless_)
      {
        if (point_count_data_vector_iterator_ == point_count_data_vector_.end ())
        {
          PCL_WARN ("[pcl::io::OctreePointCloudCompression::deserializeTreeCallback] The stream holds fewer point counts than the octree has leaves, skipping leaf.\n");
          return;
        }

        pointCount = *point_count_data_vector_iterator_++;
        const auto cloudSize = output_->size ();
        output_->points.resize (cloudSize + pointCount);
        for (std::size_t i = 0; i < pointCount; ++i)
        {
          if (!decodeLosslessPoint (output_->points[cloudSize + i]))
            PCL_WARN ("[pcl::io::OctreePointCloudCompression::deserializeTreeCallback] Not enough lossless point data left in the stream.\n");
        }
        return;
      }

      if (!do_voxel_grid_enDecoding_)
      {
        // get current cloud size
        const auto cloudSize = output_->size ();

        // the stream may hold fewer point counts than the tree has leaves
        if (point_count_data_vector_iterator_ == point_count_data_vector_.end ())
        {
          PCL_WARN ("[pcl::io::OctreePointCloudCompression::deserializeTreeCallback] The stream holds fewer point counts than the octree has leaves, skipping leaf.\n");
          return;
        }

        // get amount of point to be decoded
        pointCount = *point_count_data_vector_iterator_;
        point_count_data_vector_iterator_++;

        // increase point cloud by amount of voxel points
        for (std::size_t i = 0; i < pointCount; i++)
          output_->points.push_back (newPoint);

        // calculate position of lower voxel corner
        double lowerVoxelCorner[3];
        lowerVoxelCorner[0] = static_cast<double> (key_arg.x) * this->resolution_ + this->min_x_;
        lowerVoxelCorner[1] = static_cast<double> (key_arg.y) * this->resolution_ + this->min_y_;
        lowerVoxelCorner[2] = static_cast<double> (key_arg.z) * this->resolution_ + this->min_z_;

        // decode differentially encoded points
        point_coder_.decodePoints (output_, lowerVoxelCorner, cloudSize, cloudSize + pointCount);
      }
      else
      {
        // calculate center of lower voxel corner
        newPoint.x = static_cast<float> ((static_cast<double> (key_arg.x) + 0.5) * this->resolution_ + this->min_x_);
        newPoint.y = static_cast<float> ((static_cast<double> (key_arg.y) + 0.5) * this->resolution_ + this->min_y_);
        newPoint.z = static_cast<float> ((static_cast<double> (key_arg.z) + 0.5) * this->resolution_ + this->min_z_);

        // add point to point cloud
        output_->points.push_back (newPoint);
      }

      if (cloud_with_color_)
      {
        if (data_with_color_)
          // decode color information
          color_coder_.decodePoints (output_, output_->size () - pointCount,
                                    output_->size (), point_color_offset_);
        else
          // set default color information
          color_coder_.setDefaultColor (output_, output_->size () - pointCount,
                                       output_->size (), point_color_offset_);
      }

      if (data_with_normals_)
      {
        if (do_voxel_grid_enDecoding_)
        {
          decodeNormal (output_->points[output_->size () - pointCount]);
        }
        else
        {
          const std::size_t begin = output_->size () - pointCount;
          for (std::size_t i = 0; i < pointCount; ++i)
            decodeNormal (output_->points[begin + i]);
        }
      }
    }

    //////////////////////////////////////////////////////////////////////////////////////////////
    template<typename PointT, typename LeafT, typename BranchT, typename OctreeT> void
    OctreePointCloudCompression<PointT, LeafT, BranchT, OctreeT>::analyzePointFields ()
    {
      point_fields_ = pcl::getFields<PointT> ();
      cloud_with_color_ = false;
      cloud_with_normals_ = false;
      point_color_offset_ = 0;
      normal_field_size_ = 0;
      lossless_point_record_size_ = 0;

      for (const auto& field : point_fields_)
        lossless_point_record_size_ += static_cast<std::size_t> (pcl::getFieldSize (field.datatype)) * field.count;

      int color_index = pcl::getFieldIndex<PointT> ("rgb", point_fields_);
      if (color_index == -1)
        color_index = pcl::getFieldIndex<PointT> ("rgba", point_fields_);
      if (color_index >= 0)
      {
        const auto& color_field = point_fields_[color_index];
        const auto color_size = static_cast<std::size_t> (pcl::getFieldSize (color_field.datatype)) * color_field.count;
        if (color_size == sizeof (std::uint32_t))
        {
          point_color_offset_ = static_cast<unsigned char> (color_field.offset);
          cloud_with_color_ = true;
        }
      }

      const char* normal_names[3] = {"normal_x", "normal_y", "normal_z"};
      bool has_normals = true;
      for (int component = 0; component < 3; ++component)
      {
        const int normal_index = pcl::getFieldIndex<PointT> (normal_names[component], point_fields_);
        if (normal_index < 0)
        {
          has_normals = false;
          break;
        }

        const auto& normal_field = point_fields_[normal_index];
        const auto normal_size = static_cast<std::size_t> (pcl::getFieldSize (normal_field.datatype)) * normal_field.count;
        if (normal_field.datatype != pcl::PCLPointField::FLOAT32 || normal_size != sizeof (float))
        {
          has_normals = false;
          break;
        }
        normal_offsets_[component] = static_cast<unsigned char> (normal_field.offset);
      }

      cloud_with_normals_ = has_normals;
      normal_field_size_ = has_normals ? static_cast<unsigned char> (sizeof (float)) : 0;
    }

    //////////////////////////////////////////////////////////////////////////////////////////////
    template<typename PointT, typename LeafT, typename BranchT, typename OctreeT> void
    OctreePointCloudCompression<PointT, LeafT, BranchT, OctreeT>::encodeLosslessPoint (const PointT& point_arg)
    {
      const char* point_data = reinterpret_cast<const char*> (&point_arg);
      for (const auto& field : point_fields_)
      {
        const auto field_size = static_cast<std::size_t> (pcl::getFieldSize (field.datatype)) * field.count;
        lossless_point_data_vector_.insert (lossless_point_data_vector_.end (),
                                            point_data + field.offset,
                                            point_data + field.offset + field_size);
      }
    }

    //////////////////////////////////////////////////////////////////////////////////////////////
    template<typename PointT, typename LeafT, typename BranchT, typename OctreeT> bool
    OctreePointCloudCompression<PointT, LeafT, BranchT, OctreeT>::decodeLosslessPoint (PointT& point_arg)
    {
      char* point_data = reinterpret_cast<char*> (&point_arg);
      for (const auto& field : point_fields_)
      {
        const auto field_size = static_cast<std::size_t> (pcl::getFieldSize (field.datatype)) * field.count;
        if (lossless_point_data_position_ + field_size > lossless_point_data_vector_.size ())
          return false;

        std::memcpy (point_data + field.offset,
                     lossless_point_data_vector_.data () + lossless_point_data_position_,
                     field_size);
        lossless_point_data_position_ += field_size;
      }
      return true;
    }

    //////////////////////////////////////////////////////////////////////////////////////////////
    template<typename PointT, typename LeafT, typename BranchT, typename OctreeT> void
    OctreePointCloudCompression<PointT, LeafT, BranchT, OctreeT>::encodeNormal (const PointT& point_arg)
    {
      const char* point_data = reinterpret_cast<const char*> (&point_arg);
      for (int component = 0; component < 3; ++component)
      {
        normal_data_vector_.insert (normal_data_vector_.end (),
                                    point_data + normal_offsets_[component],
                                    point_data + normal_offsets_[component] + normal_field_size_);
      }
    }

    //////////////////////////////////////////////////////////////////////////////////////////////
    template<typename PointT, typename LeafT, typename BranchT, typename OctreeT> bool
    OctreePointCloudCompression<PointT, LeafT, BranchT, OctreeT>::decodeNormal (PointT& point_arg)
    {
      constexpr std::size_t normal_size = 3 * sizeof (float);
      if (normal_data_position_ + normal_size > normal_data_vector_.size ())
      {
        PCL_WARN ("[pcl::io::OctreePointCloudCompression::decodeNormal] Not enough normal data left in the stream, skipping normals.\n");
        return false;
      }

      if (cloud_with_normals_)
      {
        char* point_data = reinterpret_cast<char*> (&point_arg);
        for (int component = 0; component < 3; ++component)
        {
          std::memcpy (point_data + normal_offsets_[component],
                       normal_data_vector_.data () + normal_data_position_ + component * sizeof (float),
                       sizeof (float));
        }
      }
      normal_data_position_ += normal_size;
      return true;
    }

    //////////////////////////////////////////////////////////////////////////////////////////////
    template<typename PointT, typename LeafT, typename BranchT, typename OctreeT> void
    OctreePointCloudCompression<PointT, LeafT, BranchT, OctreeT>::restoreOriginalPointOrder ()
    {
      if (point_index_data_vector_.size () != output_->size ())
      {
        PCL_WARN ("[pcl::io::OctreePointCloudCompression::restoreOriginalPointOrder] Point index count does not match decoded cloud size.\n");
        return;
      }

      PointCloud ordered_cloud;
      ordered_cloud.points.resize (output_->size ());
      std::vector<bool> seen (output_->size (), false);
      for (std::size_t i = 0; i < output_->size (); ++i)
      {
        const auto index = point_index_data_vector_[i];
        if (index >= output_->size () || seen[index])
        {
          PCL_WARN ("[pcl::io::OctreePointCloudCompression::restoreOriginalPointOrder] Invalid point index in stream.\n");
          return;
        }
        ordered_cloud.points[index] = output_->points[i];
        seen[index] = true;
      }

      output_->points.swap (ordered_cloud.points);
    }
  }
}

#endif

