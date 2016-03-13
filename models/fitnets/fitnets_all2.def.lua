-- Model: fitnets_all2.def.lua
-- Description: Student network for FitNets branch
-- Input size: 3x96x96
-- Components: Mostly `nn`
-- Devices: CPU and CUDA
--
-- Licensed under the Apache License, Version 2.0 (the "License");
-- you may not use this file except in compliance with the License.
-- You may obtain a copy of the License at
--
--     http://www.apache.org/licenses/LICENSE-2.0
--
-- Unless required by applicable law or agreed to in writing, software
-- distributed under the License is distributed on an "AS IS" BASIS,
-- WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
-- See the License for the specific language governing permissions and
-- limitations under the License.

imgDim = 96
-- 624480 parameters
function createModel()
local net = nn.Sequential()
   local depths = torch.Tensor{8, 16, 32, 64, 64,
                              64, 96, 128, 192}


   -- Layer 1: Convolution with filter size 7, step size 2, padding 3
   -- Output size: 48 x 48 x depths[1] (width, height, depth)
   net:add(nn.SpatialConvolutionMM(3, depths[1], 7, 7, 2, 2, 3, 3))
   net:add(nn.SpatialBatchNormalization(depths[1]))
   net:add(nn.ReLU())

   -- Layer 4: Convolution with filter size 7, step size 2, padding 3
   -- Output size: 24 x 24 x depths[2] (width, height, depth)
   net:add(nn.SpatialConvolutionMM(depths[1], depths[2], 7, 7, 2, 2, 3, 3))
   net:add(nn.SpatialBatchNormalization(depths[2]))
   net:add(nn.ReLU())

   -- Layer 7: Convolution with filter size 5, step size 1, padding 2
   -- Output size: 24 x 24 x depths[3] (width, height, depth)
   net:add(nn.SpatialConvolutionMM(depths[2], depths[3], 5, 5, 1, 1, 2, 2))
   net:add(nn.SpatialBatchNormalization(depths[3]))
   net:add(nn.ReLU())

   -- Layer 10: Convolution with filter size 5, step size 2, padding 2
   -- Output size: 12 x 12 x depths[4] (width, height, depth)
   net:add(nn.SpatialConvolutionMM(depths[3], depths[4], 5, 5, 2, 2, 2, 2))
   net:add(nn.SpatialBatchNormalization(depths[4]))
   net:add(nn.ReLU())

   -- Layer 13: Convolution with filter size 5, step size 2, padding 2
   -- Output size: 6 x 6 x depths[5] (width, height, depth)
   net:add(nn.SpatialConvolutionMM(depths[4], depths[5], 5, 5, 2, 2, 2, 2))
   net:add(nn.SpatialBatchNormalization(depths[5]))
   net:add(nn.ReLU())

   -- Layer 16: Convolution with filter size 3, step size 1, padding 1
   -- Output size: 6 x 6 x depths[6] (width, height, depth)
   net:add(nn.SpatialConvolutionMM(depths[5], depths[6], 3, 3, 1, 1, 1, 1))
   net:add(nn.SpatialBatchNormalization(depths[6]))
   net:add(nn.ReLU())

   -- Layer 19: Convolution with filter size 3, step size 1, padding 1
   -- Output size: 6 x 6 x depths[7] (width, height, depth)
   net:add(nn.SpatialConvolutionMM(depths[6], depths[7], 3, 3, 1, 1, 1, 1))
   net:add(nn.SpatialBatchNormalization(depths[7]))
   net:add(nn.ReLU())

   -- Layer 22: Convolution with filter size 3, step size 2, padding 1
   -- Output size: 3 x 3 x depths[8] (width, height, depth)
   net:add(nn.SpatialConvolutionMM(depths[7], depths[8], 3, 3, 2, 2, 1, 1))
   net:add(nn.SpatialBatchNormalization(depths[8]))
   net:add(nn.ReLU())

   -- Layer 22: Convolution with filter size 3, step size 1, padding 1
   -- Output size: 3 x 3 x depths[9] (width, height, depth)
   net:add(nn.SpatialConvolutionMM(depths[8], depths[9], 3, 3, 1, 1, 1, 1))
   net:add(nn.SpatialBatchNormalization(depths[9]))
   net:add(nn.ReLU())

   net:add(nn.SpatialAveragePooling(3, 3))

   net:add(nn.View(depths[-1]))
   net:add(nn.Linear(depths[-1], opt.embSize))
   net:add(nn.Normalize(2))

   return net
end
