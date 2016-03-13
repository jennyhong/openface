-- Model: fitnets1.def.lua
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

function createModel()
   local net = nn.Sequential()
   local depths = torch.Tensor{8, 16, 24, 28, 32, 36}

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

   -- Layer 10: Convolution with filter size 5, step size 1, padding 2
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

   -- REGRESSOR LAYER
   net:add(nn.SpatialConvolutionMM(depths[6], 640, 1, 1, 1, 1, 0, 0))

   -- net:add(nn.Reshape(324))

   -- 6 * 6 * 640 = 23040
   -- 3 * 3 * 36 = 324
   -- net:add(nn.Linear(324, 23040))

   return net
end
